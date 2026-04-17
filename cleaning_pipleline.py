"""
cleaning_pipeline.py
EventOracle – Data Cleaning & Unified Feature Pipeline (Person 7)

Reads from four upstream MongoDB collections:
  - prices   (Person 3) → OHLCV + technical indicators
  - news     (Person 4) → headlines, sentiment, event tags, velocity
  - events   (Person 5) → macro events with actual/consensus/previous
  - flows    (Person 6) → FII/DII net flows + momentum features

Produces:
  - Collection: `features`  (one row per asset per trading day)
  - 50+ engineered features ready for ML Stage 1 models
  - SHAP-ready flat schema with lag features and anomaly flags

Usage:
    python cleaning_pipeline.py                         # Full run, all assets
    python cleaning_pipeline.py --asset NIFTY           # Single asset
    python cleaning_pipeline.py --start 2024-01-01      # Custom date range
    python cleaning_pipeline.py --dry-run               # Skip MongoDB write

Scheduling (cron – daily at 9:00 AM IST / 3:30 AM UTC, after all upstream):
    30 3 * * * /usr/bin/python3 /path/to/cleaning_pipeline.py >> /var/log/eventoracle_cleaning.log 2>&1

Dependencies:
    pip install pymongo pandas numpy python-dotenv pytz
"""

import os
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.cleaning")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = "eventoracle"
IST         = pytz.timezone("Asia/Kolkata")
BATCH_SIZE  = 500

# Source collections (read-only)
COL_PRICES  = "prices"
COL_NEWS    = "news"
COL_EVENTS  = "events"
COL_FLOWS   = "flows"

# Output collection
COL_FEATURES = "features"

# All tracked assets (must match Person 3 asset names)
ALL_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]

# Event categories for one-hot encoding (matches Person 4 tagging)
EVENT_CATEGORIES = [
    "RBI", "Fed", "CPI", "Oil", "Gold", "Silver",
    "Platinum", "Bitcoin", "Nifty", "Currency",
    "FII", "GDP/IIP", "SEBI", "Geopolitical", "Other",
]

# Lag windows for time-series features
LAG_WINDOWS = [1, 3, 5]

# Forward-fill limit for macro events (days)
EVENT_FFILL_LIMIT = 3

# Anomaly thresholds
NEWS_SPIKE_THRESHOLD_STD = 2.0
FII_ANOMALY_ZSCORE = 2.0


# ─────────────────────────────────────────────
# 1. DATABASE CONNECTION
# ─────────────────────────────────────────────

def get_mongo_db():
    """Connect to MongoDB Atlas; return (client, db) tuple."""
    if not MONGO_URI:
        raise ValueError(
            "MONGO_URI not set. Add it to your .env file:\n"
            "  MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/"
        )
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    log.info("✅ Connected to MongoDB Atlas.")
    return client, client[DB_NAME]


def ensure_features_index(db) -> None:
    """Create unique compound index on the features collection."""
    col = db[COL_FEATURES]
    col.create_index(
        [("timestamp", 1), ("asset", 1)],
        unique=True,
        background=True,
        name="features_timestamp_asset_unique",
    )
    log.info(f"[MONGO] Index ensured on '{COL_FEATURES}'.")


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────

def _to_ist_date(ts) -> Optional[datetime]:
    """
    Normalize any timestamp to a timezone-aware IST datetime
    truncated to midnight (date-level granularity).
    Returns None if conversion fails.
    """
    try:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, utc=True)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_ist = ts.astimezone(IST)
        return ts_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        return None


def load_prices(db, assets: list, start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load OHLCV + technical features from the prices collection.
    Returns one row per (timestamp, asset).
    """
    log.info("[LOAD] Loading prices...")
    query = {"asset": {"$in": assets}}
    if start_date:
        query["timestamp"] = {"$gte": start_date}

    cursor = db[COL_PRICES].find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        log.warning("[LOAD] prices collection returned no data.")
        return df

    df["timestamp"] = df["timestamp"].apply(_to_ist_date)
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "asset"])
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    log.info(f"[LOAD] prices → {len(df)} rows across {df['asset'].nunique()} assets.")
    return df


def load_news(db, start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load all news records. Returns one row per article with
    normalized IST timestamp and cleaned headline.
    """
    log.info("[LOAD] Loading news...")
    query = {}
    if start_date:
        query["timestamp"] = {"$gte": start_date}

    cursor = db[COL_NEWS].find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        log.warning("[LOAD] news collection returned no data.")
        return df

    df["timestamp"] = df["timestamp"].apply(_to_ist_date)
    df = df.dropna(subset=["timestamp"])

    # Deduplicate: keep highest sentiment_score when headline + timestamp collide
    if "headline" in df.columns and "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
        df = (
            df.sort_values("sentiment_score", ascending=False)
              .drop_duplicates(subset=["headline", "timestamp"])
        )

    log.info(f"[LOAD] news → {len(df)} articles after dedup.")
    return df


def load_events(db, start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load macro economic events.
    Returns one row per (event_name, country, timestamp).
    """
    log.info("[LOAD] Loading events...")
    query = {}
    if start_date:
        query["timestamp"] = {"$gte": start_date}

    cursor = db[COL_EVENTS].find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        log.warning("[LOAD] events collection returned no data.")
        return df

    df["timestamp"] = df["timestamp"].apply(_to_ist_date)
    df = df.dropna(subset=["timestamp"])

    # Deduplicate on event_name + country + timestamp
    df = df.drop_duplicates(subset=["event_name", "country", "timestamp"])

    log.info(f"[LOAD] events → {len(df)} records after dedup.")
    return df


def load_flows(db, start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load FII/DII institutional flow data.
    Returns one row per trading day (no asset dimension – market-wide).
    """
    log.info("[LOAD] Loading flows...")
    query = {}
    if start_date:
        query["timestamp"] = {"$gte": start_date}

    cursor = db[COL_FLOWS].find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        log.warning("[LOAD] flows collection returned no data.")
        return df

    df["timestamp"] = df["timestamp"].apply(_to_ist_date)
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info(f"[LOAD] flows → {len(df)} trading days.")
    return df


# ─────────────────────────────────────────────
# 3. CLEANING FUNCTIONS
# ─────────────────────────────────────────────

def clean_news(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw news articles into daily features:
      - avg_sentiment, sentiment_std
      - news_volume (article count)
      - news_velocity (articles per hour, approximated daily)
      - event_type one-hot counts
      - news_spike_flag (volume anomaly)
    Returns one row per calendar day.
    """
    if df_news.empty:
        log.warning("[CLEAN] News dataframe is empty; skipping.")
        return pd.DataFrame()

    log.info("[CLEAN] Aggregating news to daily features...")

    df = df_news.copy()
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score", 0), errors="coerce").fillna(0.0)

    # Event type one-hot
    if "event_type" in df.columns:
        for cat in EVENT_CATEGORIES:
            col_name = f"news_event_{cat.lower().replace('/', '_')}"
            df[col_name] = (df["event_type"] == cat).astype(int)
    else:
        for cat in EVENT_CATEGORIES:
            df[f"news_event_{cat.lower().replace('/', '_')}"] = 0

    event_cols = [f"news_event_{c.lower().replace('/', '_')}" for c in EVENT_CATEGORIES]

    # Build aggregation dict
    agg_dict = {
        "avg_sentiment":    ("sentiment_score", "mean"),
        "sentiment_std":    ("sentiment_score", "std"),
        "news_volume":      ("sentiment_score", "count"),
    }
    for col in event_cols:
        agg_dict[col] = (col, "sum")

    daily = df.groupby("timestamp").agg(**agg_dict).reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)

    # news_velocity: articles per hour (assuming ~16 trading hours coverage)
    daily["news_velocity"] = (daily["news_volume"] / 16.0).round(4)

    # News spike anomaly flag
    vol_mean = daily["news_volume"].mean()
    vol_std  = daily["news_volume"].std()
    threshold = vol_mean + NEWS_SPIKE_THRESHOLD_STD * (vol_std if vol_std > 0 else 1.0)
    daily["news_spike_flag"] = (daily["news_volume"] > threshold).astype(int)

    # Round floats
    daily["avg_sentiment"] = daily["avg_sentiment"].round(6)
    daily["sentiment_std"] = daily["sentiment_std"].round(6)

    log.info(f"[CLEAN] news → {len(daily)} daily rows | "
             f"spikes: {daily['news_spike_flag'].sum()}")
    return daily.sort_values("timestamp").reset_index(drop=True)


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer event-level features:
      - surprise = actual - consensus
      - normalized_surprise = surprise / |previous|
      - event_flag (binary)
      - event category one-hot columns
    Returns one row per (timestamp, event_name, country) with added features.
    """
    if df_events.empty:
        log.warning("[CLEAN] Events dataframe is empty; skipping.")
        return pd.DataFrame()

    log.info("[CLEAN] Engineering event features...")

    df = df_events.copy()

    for col in ["actual", "consensus", "previous"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    df["surprise"] = (df["actual"] - df["consensus"]).round(6)
    df["normalized_surprise"] = (
        df["surprise"] / df["previous"].abs().replace(0, np.nan)
    ).round(6)
    df["event_flag"] = 1

    # Map event_name → EVENT_CATEGORIES for one-hot
    event_name_map = {
        "CPI":      "CPI",
        "IIP":      "GDP/IIP",
        "GDP":      "GDP/IIP",
        "FED_RATE": "Fed",
        "RBI_REPO": "RBI",
    }
    df["_ev_cat"] = df["event_name"].map(event_name_map).fillna("Other")

    for cat in EVENT_CATEGORIES:
        col_name = f"event_cat_{cat.lower().replace('/', '_')}"
        df[col_name] = (df["_ev_cat"] == cat).astype(int)

    df = df.drop(columns=["_ev_cat"])
    log.info(f"[CLEAN] events → {len(df)} records with surprise features.")
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(6)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean().round(6)


def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all price-derived features exist per asset.
    Adds any missing indicators that Person 3 may not have computed.
    Also adds lag features for t-1, t-3, t-5.
    """
    result_frames = []

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        # Core price features (add if missing)
        if "returns" not in grp.columns or grp["returns"].isna().all():
            grp["returns"] = grp["close"].pct_change().round(6)
        if "rsi" not in grp.columns or grp["rsi"].isna().all():
            grp["rsi"] = _rsi(grp["close"], 14)
        if "atr" not in grp.columns or grp["atr"].isna().all():
            grp["atr"] = _atr(grp, 14)
        if "rolling_volatility" not in grp.columns or grp["rolling_volatility"].isna().all():
            grp["rolling_volatility"] = grp["returns"].rolling(10).std().round(6)
        if "volume_zscore" not in grp.columns or grp["volume_zscore"].isna().all():
            vol_mean = grp["volume"].rolling(20).mean()
            vol_std  = grp["volume"].rolling(20).std().replace(0, np.nan)
            grp["volume_zscore"] = ((grp["volume"] - vol_mean) / vol_std).round(6)

        # Extended price features
        grp["hl_range"]        = ((grp["high"] - grp["low"]) / grp["close"].replace(0, np.nan)).round(6)
        grp["close_to_high"]   = ((grp["close"] - grp["low"])  / (grp["high"] - grp["low"]).replace(0, np.nan)).round(6)
        grp["rolling_mean_5"]  = grp["close"].rolling(5).mean().round(4)
        grp["rolling_mean_20"] = grp["close"].rolling(20).mean().round(4)
        grp["price_vs_ma5"]    = ((grp["close"] - grp["rolling_mean_5"])  / grp["rolling_mean_5"].replace(0, np.nan)).round(6)
        grp["price_vs_ma20"]   = ((grp["close"] - grp["rolling_mean_20"]) / grp["rolling_mean_20"].replace(0, np.nan)).round(6)

        # Lag features for key signals
        lag_targets = ["returns", "rsi", "rolling_volatility", "volume_zscore"]
        for col in lag_targets:
            if col in grp.columns:
                for lag in LAG_WINDOWS:
                    grp[f"{col}_lag{lag}"] = grp[col].shift(lag).round(6)

        result_frames.append(grp)

    if not result_frames:
        return df

    out = pd.concat(result_frames, ignore_index=True)
    log.info(f"[FEATURES] Price features engineered. Shape: {out.shape}")
    return out


def engineer_flow_features(df_flows: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance flow features:
      - flow_to_price_ratio (FII net / NIFTY close)
      - fii_dominance (fii_net / (|fii_net| + |dii_net|))
      - Lag features for FII/DII flows
      - Extreme flow anomaly flag (if not already present from Person 6)
    Returns the enriched flows dataframe (still one row per day).
    """
    if df_flows.empty:
        return df_flows

    df = df_flows.copy()

    # Ensure numeric
    for col in ["fii_net", "dii_net"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # flow_to_price_ratio: join NIFTY close price
    if not df_prices.empty and "fii_net" in df.columns:
        nifty = df_prices[df_prices["asset"] == "NIFTY"][["timestamp", "close"]].copy()
        nifty = nifty.rename(columns={"close": "nifty_close"})
        df = df.merge(nifty, on="timestamp", how="left")
        df["flow_to_price_ratio"] = (
            df["fii_net"] / df["nifty_close"].replace(0, np.nan)
        ).round(6)
    else:
        df["flow_to_price_ratio"] = np.nan

    # FII dominance ratio
    if "fii_net" in df.columns and "dii_net" in df.columns:
        total_abs = df["fii_net"].abs() + df["dii_net"].abs()
        df["fii_dominance"] = (df["fii_net"] / total_abs.replace(0, np.nan)).round(6)

    # Anomaly flag (recalculate if not present)
    if "flow_anomaly" not in df.columns and "fii_net" in df.columns:
        roll_mean = df["fii_net"].rolling(20).mean()
        roll_std  = df["fii_net"].rolling(20).std().replace(0, np.nan)
        z = (df["fii_net"] - roll_mean) / roll_std
        df["flow_anomaly"] = (z.abs() > FII_ANOMALY_ZSCORE).astype(int)

    # Lag features for FII/DII
    for col in ["fii_net", "dii_net"]:
        if col in df.columns:
            for lag in LAG_WINDOWS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag).round(4)

    log.info(f"[FEATURES] Flow features enriched. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. MERGING LOGIC
# ─────────────────────────────────────────────

def merge_features(
    df_prices: pd.DataFrame,
    df_flows: pd.DataFrame,
    df_events: pd.DataFrame,
    df_news_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all feature tables onto the prices base (one row per asset per day).

    Join order:
        prices → flows → events (aggregated daily) → news (aggregated daily)

    Filling strategy:
        - sentiment, event flags → 0
        - flows → 0
        - event surprise → NaN (no synthetic data)
        - event forward-fill limited to EVENT_FFILL_LIMIT days
    """
    log.info("[MERGE] Merging all feature tables onto prices base...")

    if df_prices.empty:
        log.error("[MERGE] Prices dataframe is empty — cannot merge.")
        return pd.DataFrame()

    base = df_prices.copy()

    # ── 1. Merge Flows ────────────────────────────────────────
    if not df_flows.empty:
        flow_cols = [c for c in df_flows.columns if c != "timestamp"]
        base = base.merge(df_flows[["timestamp"] + flow_cols], on="timestamp", how="left")
        flow_fill_cols = [c for c in flow_cols if c not in ("flow_anomaly",)]
        base[flow_fill_cols] = base[flow_fill_cols].fillna(0.0)
        if "flow_anomaly" in base.columns:
            base["flow_anomaly"] = base["flow_anomaly"].fillna(0).astype(int)
        log.info(f"[MERGE] After flows join: {base.shape}")
    else:
        log.warning("[MERGE] No flows data; flow columns will be absent.")

    # ── 2. Aggregate Events to Daily Level ───────────────────
    if not df_events.empty:
        # Pivot surprise metrics: one column per (event_name, country) pair
        event_pivot_rows = []
        for ts, grp in df_events.groupby("timestamp"):
            row = {"timestamp": ts, "event_flag": 1}

            for _, ev_row in grp.iterrows():
                prefix = f"ev_{ev_row['event_name'].lower()}_{ev_row.get('country', 'na').lower()}"
                row[f"{prefix}_actual"]    = ev_row.get("actual")
                row[f"{prefix}_surprise"]  = ev_row.get("surprise")
                row[f"{prefix}_norm_surp"] = ev_row.get("normalized_surprise")

            # Aggregate event category one-hots (OR across events on same day)
            for cat in EVENT_CATEGORIES:
                col = f"event_cat_{cat.lower().replace('/', '_')}"
                if col in grp.columns:
                    row[col] = int(grp[col].max())

            event_pivot_rows.append(row)

        df_ev_daily = pd.DataFrame(event_pivot_rows)
        df_ev_daily = df_ev_daily.sort_values("timestamp")

        # Forward-fill event flags for limited window (e.g. multi-day impact)
        # Merge onto base
        base = base.merge(df_ev_daily, on="timestamp", how="left")

        # Forward-fill surprise columns within limited window
        ev_surprise_cols = [c for c in base.columns if "surprise" in c or "norm_surp" in c]
        base[ev_surprise_cols] = (
            base.sort_values("timestamp")[ev_surprise_cols]
                .fillna(method="ffill", limit=EVENT_FFILL_LIMIT)
        )
        base["event_flag"] = base["event_flag"].fillna(0).astype(int)

        # Fill category one-hots with 0
        ev_cat_cols = [c for c in base.columns if c.startswith("event_cat_")]
        base[ev_cat_cols] = base[ev_cat_cols].fillna(0).astype(int)

        log.info(f"[MERGE] After events join: {base.shape}")
    else:
        log.warning("[MERGE] No events data; event columns will be absent.")
        base["event_flag"] = 0

    # ── 3. Merge News ─────────────────────────────────────────
    if not df_news_daily.empty:
        base = base.merge(df_news_daily, on="timestamp", how="left")

        # Fill sentiment with neutral; volume/velocity with 0
        base["avg_sentiment"]  = base.get("avg_sentiment", pd.Series(dtype=float)).fillna(0.0)
        base["sentiment_std"]  = base.get("sentiment_std",  pd.Series(dtype=float)).fillna(0.0)
        base["news_volume"]    = base.get("news_volume",    pd.Series(dtype=float)).fillna(0).astype(int)
        base["news_velocity"]  = base.get("news_velocity",  pd.Series(dtype=float)).fillna(0.0)
        base["news_spike_flag"] = base.get("news_spike_flag", pd.Series(dtype=float)).fillna(0).astype(int)

        news_event_cols = [c for c in base.columns if c.startswith("news_event_")]
        base[news_event_cols] = base[news_event_cols].fillna(0).astype(int)

        log.info(f"[MERGE] After news join: {base.shape}")
    else:
        log.warning("[MERGE] No news data; news columns will be absent.")
        base["avg_sentiment"] = 0.0
        base["news_volume"]   = 0
        base["news_spike_flag"] = 0

    # ── 4. Add Combined Anomaly Flag ──────────────────────────
    anomaly_conditions = pd.Series(False, index=base.index)
    if "news_spike_flag" in base.columns:
        anomaly_conditions |= base["news_spike_flag"].astype(bool)
    if "flow_anomaly" in base.columns:
        anomaly_conditions |= base["flow_anomaly"].astype(bool)
    base["combined_anomaly_flag"] = anomaly_conditions.astype(int)

    base = base.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    log.info(f"[MERGE] Final merged shape: {base.shape}")
    return base


# ─────────────────────────────────────────────
# 6. POST-MERGE FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-dataset interaction features that require the merged table:
      - sentiment_x_returns  (news sentiment × price return)
      - fii_x_returns        (FII flow direction × return)
      - event_x_volatility   (event day × rolling volatility)
      - surprise_x_rsi       (event surprise magnitude × RSI level)
      - Lag features for sentiment and flows (t-1, t-3, t-5)
    """
    log.info("[FEATURES] Adding cross-dataset interaction features...")
    result_frames = []

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        # Interaction features
        if "avg_sentiment" in grp.columns and "returns" in grp.columns:
            grp["sentiment_x_returns"] = (grp["avg_sentiment"] * grp["returns"]).round(8)

        if "fii_net" in grp.columns and "returns" in grp.columns:
            fii_sign = np.sign(grp["fii_net"].fillna(0))
            ret_sign = np.sign(grp["returns"].fillna(0))
            grp["fii_return_alignment"] = (fii_sign == ret_sign).astype(int)

        if "event_flag" in grp.columns and "rolling_volatility" in grp.columns:
            grp["event_x_volatility"] = (grp["event_flag"] * grp["rolling_volatility"]).round(8)

        # Find any surprise column for the first event type available
        surprise_cols = [c for c in grp.columns if c.endswith("_surprise") and not c.endswith("norm_surp")]
        if surprise_cols and "rsi" in grp.columns:
            combined_surprise = grp[surprise_cols].fillna(0).abs().max(axis=1)
            grp["surprise_x_rsi"] = (combined_surprise * grp["rsi"].fillna(50)).round(6)

        # Lag features for news + flow signals
        lag_cross_targets = ["avg_sentiment", "fii_net", "news_volume"]
        for col in lag_cross_targets:
            if col in grp.columns:
                for lag in LAG_WINDOWS:
                    lag_col = f"{col}_lag{lag}"
                    if lag_col not in grp.columns:   # don't duplicate
                        grp[lag_col] = grp[col].shift(lag).round(6)

        result_frames.append(grp)

    if not result_frames:
        return df

    out = pd.concat(result_frames, ignore_index=True)
    log.info(f"[FEATURES] Cross features added. Final shape: {out.shape}")
    return out


# ─────────────────────────────────────────────
# 7. FINAL CLEANING & SCHEMA ENFORCEMENT
# ─────────────────────────────────────────────

def finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass:
      - Round all floats to 6 decimal places
      - Convert int columns
      - Replace inf/-inf with NaN
      - Ensure no pandas-specific types remain (for BSON compatibility)
    """
    log.info("[SCHEMA] Finalizing schema...")

    df = df.replace([np.inf, -np.inf], np.nan)

    int_cols = [
        "volume", "news_volume", "event_flag", "news_spike_flag",
        "flow_anomaly", "combined_anomaly_flag",
    ] + [c for c in df.columns if c.startswith("news_event_") or c.startswith("event_cat_")]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = df.select_dtypes(include=[float]).columns.tolist()
    for col in float_cols:
        df[col] = df[col].round(6)

    # Ensure timestamp is Python datetime (not pandas Timestamp) — done at upload
    log.info(f"[SCHEMA] Final feature count: {len(df.columns)} columns | {len(df)} rows.")
    return df.sort_values(["asset", "timestamp"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# 8. DATABASE STORAGE
# ─────────────────────────────────────────────

def save_to_mongo(df: pd.DataFrame, db, dry_run: bool = False) -> int:
    """
    Upsert all feature rows into the `features` collection.
    Returns count of newly inserted documents.
    """
    if df.empty:
        log.warning("[MONGO] Nothing to save — dataframe is empty.")
        return 0

    records = df.to_dict("records")

    # Serialize for MongoDB BSON
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        for k, v in list(rec.items()):
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)

    if dry_run:
        log.info(f"[MONGO] DRY RUN — would upsert {len(records)} records into '{COL_FEATURES}'.")
        return 0

    ops = [
        UpdateOne(
            {"timestamp": rec["timestamp"], "asset": rec["asset"]},
            {"$set": rec},   # $set (not $setOnInsert) so features are refreshed on re-run
            upsert=True,
        )
        for rec in records
    ]

    total_upserted = 0
    col = db[COL_FEATURES]

    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = col.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count + result.modified_count
        except BulkWriteError as bwe:
            total_upserted += bwe.details.get("nUpserted", 0) + bwe.details.get("nModified", 0)
            log.debug(f"[MONGO] BulkWriteError details: {bwe.details}")

    log.info(f"[MONGO] ✅ Upserted/updated {total_upserted} records in '{COL_FEATURES}'.")
    return total_upserted


# ─────────────────────────────────────────────
# 9. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(
    assets: list = None,
    start_date: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Full cleaning + feature unification pipeline.

    Args:
        assets     : List of asset names to process (default: ALL_ASSETS).
        start_date : ISO date string 'YYYY-MM-DD' to filter data from (default: all history).
        dry_run    : If True, skip MongoDB write (useful for testing).
    """
    if assets is None:
        assets = ALL_ASSETS

    start_dt = None
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            log.error(f"Invalid --start date '{start_date}'. Use YYYY-MM-DD format.")
            return

    log.info("=" * 65)
    log.info("EventOracle – Cleaning & Feature Pipeline (Person 7)")
    log.info(f"Assets   : {assets}")
    log.info(f"From     : {start_date or 'all history'}")
    log.info(f"Dry run  : {dry_run}")
    log.info("=" * 65)

    # ── Connect ──────────────────────────────────────────────
    try:
        client, db = get_mongo_db()
        ensure_features_index(db)
    except Exception as exc:
        log.error(f"❌ MongoDB connection failed: {exc}")
        return

    try:
        # ── Stage 1: Load ─────────────────────────────────────
        log.info("\n── Stage 1: Loading upstream collections ──")
        df_prices = load_prices(db, assets=assets, start_date=start_dt)
        df_news   = load_news(db,   start_date=start_dt)
        df_events = load_events(db, start_date=start_dt)
        df_flows  = load_flows(db,  start_date=start_dt)

        if df_prices.empty:
            log.error("❌ Prices collection is empty. Aborting pipeline.")
            return

        # ── Stage 2: Clean individual sources ─────────────────
        log.info("\n── Stage 2: Cleaning & Aggregating sources ──")
        df_news_daily  = clean_news(df_news)
        df_events_feat = clean_events(df_events)

        # ── Stage 3: Price + Flow feature engineering ─────────
        log.info("\n── Stage 3: Engineering price & flow features ──")
        df_prices_feat = engineer_price_features(df_prices)
        df_flows_feat  = engineer_flow_features(df_flows, df_prices)

        # ── Stage 4: Merge all sources ────────────────────────
        log.info("\n── Stage 4: Merging all feature tables ──")
        df_merged = merge_features(
            df_prices=df_prices_feat,
            df_flows=df_flows_feat,
            df_events=df_events_feat,
            df_news_daily=df_news_daily,
        )

        if df_merged.empty:
            log.error("❌ Merged dataframe is empty. Aborting pipeline.")
            return

        # ── Stage 5: Cross-dataset interaction features ───────
        log.info("\n── Stage 5: Cross-dataset interaction features ──")
        df_final = add_cross_features(df_merged)

        # ── Stage 6: Schema finalization ──────────────────────
        log.info("\n── Stage 6: Finalizing schema ──")
        df_final = finalize_schema(df_final)

        # ── Stage 7: Save to MongoDB ──────────────────────────
        log.info("\n── Stage 7: Saving to MongoDB ──")
        inserted = save_to_mongo(df_final, db, dry_run=dry_run)

        # ── Summary ───────────────────────────────────────────
        log.info("\n" + "=" * 65)
        log.info("  EventOracle Cleaning Pipeline – Summary")
        log.info("=" * 65)
        log.info(f"  Total rows (feature records)  : {len(df_final)}")
        log.info(f"  Features (columns)            : {len(df_final.columns)}")
        log.info(f"  Assets processed              : {df_final['asset'].nunique()}")
        if not df_final.empty:
            log.info(f"  Date range                    : "
                     f"{df_final['timestamp'].min()} → {df_final['timestamp'].max()}")
        log.info(f"  New/updated records in Mongo  : {inserted}")
        log.info(f"  Combined anomaly days         : {df_final.get('combined_anomaly_flag', pd.Series([0])).sum()}")
        log.info(f"  Dry run                       : {dry_run}")

        # Per-asset breakdown
        log.info("\n  Per-asset record counts:")
        for asset, cnt in df_final.groupby("asset").size().items():
            log.info(f"    {asset:<14} → {cnt:>5} rows")

        log.info("=" * 65)
        log.info("✅ Cleaning pipeline complete.")

    except Exception as exc:
        log.error(f"❌ Pipeline error: {exc}", exc_info=True)
    finally:
        client.close()


# ─────────────────────────────────────────────
# 10. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle – Data Cleaning & Unified Feature Pipeline (Person 7)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        choices=ALL_ASSETS,
        help="Process a single asset only (default: all assets)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only process data from this date onward (default: all history)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run all stages but skip writing to MongoDB",
    )

    args = parser.parse_args()

    selected_assets = [args.asset] if args.asset else ALL_ASSETS

    run_pipeline(
        assets=selected_assets,
        start_date=args.start,
        dry_run=args.dry_run,
    )