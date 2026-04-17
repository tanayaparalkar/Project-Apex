"""
cleaning_pipeline.py
EventOracle – Data Cleaning & Feature Unification Pipeline

Reads from MongoDB Atlas collections:
    prices  → OHLCV + technical features  (market_pipeline.py)
    flows   → FII/DII flows + features    (flows_pipeline.py)
    news    → headlines + sentiment       (news_pipeline.py)
    events  → macro events                (events_pipeline.py)

Produces a single unified "features" collection (and optional CSV) ready
for downstream modelling, with a next-day-return target variable.

Usage:
    python cleaning_pipeline.py                    # Full run
    python cleaning_pipeline.py --dry-run          # Skip MongoDB write
    python cleaning_pipeline.py --csv              # Also export features.csv
    python cleaning_pipeline.py --lookback 365     # Limit history window

Scheduling (cron – daily at 8:00 AM IST / 2:30 AM UTC, after all feed pipelines):
    30 2 * * * /usr/bin/python3 /path/to/cleaning_pipeline.py >> /var/log/eventoracle_cleaning.log 2>&1

Dependencies:
    pip install pymongo pandas numpy python-dotenv pytz
"""

import os
import logging
import argparse
from datetime import datetime, timezone, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.cleaning")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MONGO_URI       = os.getenv("MONGO_URI")
DB_NAME         = "eventoracle"

# Source collections (read-only)
PRICES_COL      = "prices"
FLOWS_COL       = "flows"
NEWS_COL        = "news"
EVENTS_COL      = "events"

# Output collection
FEATURES_COL    = "features"

IST             = pytz.timezone("Asia/Kolkata")
UTC             = pytz.utc
BATCH_SIZE      = 500

# Price asset to use as the base spine
BASE_ASSET      = "NIFTY"

# ── Event category keyword mapping ───────────────────────────
EVENT_CATEGORY_MAP = {
    "inflation":    ["cpi", "wpi", "inflation", "price index"],
    "growth":       ["gdp", "iip", "industrial production", "growth"],
    "central_bank": ["rbi", "fed", "federal reserve", "repo rate",
                     "monetary policy", "rate decision", "rate hike",
                     "rate cut", "interest rate"],
    "liquidity":    ["fii", "dii", "foreign institutional", "domestic institutional",
                     "liquidity", "open market operation", "omo"],
    "regulation":   ["sebi", "regulation", "regulatory", "circular", "compliance"],
    "geopolitical": ["war", "conflict", "sanction", "geopolit", "election",
                     "political", "tension", "crisis"],
    "oil":          ["oil", "crude", "brent", "opec", "petroleum", "energy"],
}


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def get_mongo_db():
    """Connect to MongoDB Atlas and return (client, db) tuple."""
    if not MONGO_URI:
        raise ValueError(
            "MONGO_URI is not set. "
            "Add it to your .env file: MONGO_URI=mongodb+srv://..."
        )
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    db = client[DB_NAME]
    return client, db


def ensure_indexes(db) -> None:
    """Create required indexes on the features collection."""
    col = db[FEATURES_COL]
    col.create_index(
        [("trading_date", 1)],
        unique=True,
        background=True,
        name="trading_date_unique",
    )
    log.info(f"[MONGO] Indexes ensured on '{FEATURES_COL}'.")


def to_ist_date(ts) -> Optional[date]:
    """
    Convert any timestamp-like value to an IST calendar date (Python date).
    Returns None if conversion fails.

    This is the SINGLE authoritative timestamp→date converter used everywhere
    in this pipeline to guarantee all joins happen on the same key.
    """
    if ts is None:
        return None
    try:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(IST).date()
    except Exception:
        return None


def _normalize_ts_col(series: pd.Series) -> pd.Series:
    """
    Apply to_ist_date() across an entire Series, returning a Series of
    Python date objects (not datetime) ready to be used as join keys.
    """
    return series.apply(to_ist_date)


# ─────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_prices(db, lookback_days: int = 730) -> pd.DataFrame:
    """
    Fetch NIFTY OHLCV + technical features from the prices collection.
    Only the BASE_ASSET (NIFTY) is loaded as the price spine.
    """
    log.info(f"[FETCH] Loading prices for asset='{BASE_ASSET}'…")
    since = datetime.now(UTC) - timedelta(days=lookback_days)

    docs = list(db[PRICES_COL].find(
        {"asset": BASE_ASSET, "timestamp": {"$gte": since}},
        {"_id": 0, "timestamp": 1, "close": 1, "returns": 1,
         "rsi": 1, "atr": 1, "rolling_volatility": 1},
    ))
    if not docs:
        log.warning("[FETCH] No price documents returned.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    log.info(f"[FETCH] prices → {len(df)} rows.")
    return df


def fetch_flows(db, lookback_days: int = 730) -> pd.DataFrame:
    """Fetch institutional flow features from the flows collection."""
    log.info("[FETCH] Loading flows…")
    since = datetime.now(UTC) - timedelta(days=lookback_days)

    docs = list(db[FLOWS_COL].find(
        {"timestamp": {"$gte": since}},
        {"_id": 0, "timestamp": 1, "total_flow": 1, "flow_3d": 1,
         "flow_5d": 1, "flow_momentum": 1, "flow_zscore": 1,
         "flow_anomaly": 1},
    ))
    if not docs:
        log.warning("[FETCH] No flow documents returned.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    log.info(f"[FETCH] flows → {len(df)} rows.")
    return df


def fetch_news(db, lookback_days: int = 730) -> pd.DataFrame:
    """Fetch news sentiment data from the news collection."""
    log.info("[FETCH] Loading news…")
    since = datetime.now(UTC) - timedelta(days=lookback_days)

    # Try common timestamp field names across news pipeline variants
    ts_candidates = ["published_at", "timestamp", "date"]
    sample = db[NEWS_COL].find_one()
    if sample is None:
        log.warning("[FETCH] News collection is empty.")
        return pd.DataFrame()

    ts_field = next((f for f in ts_candidates if f in sample), ts_candidates[0])
    log.debug(f"[FETCH] News timestamp field detected: '{ts_field}'")

    docs = list(db[NEWS_COL].find(
        {ts_field: {"$gte": since}},
        {"_id": 0, ts_field: 1, "title": 1, "headline": 1,
         "sentiment_score": 1},
    ))
    if not docs:
        log.warning("[FETCH] No news documents returned.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    # Normalise the timestamp column name to "timestamp"
    if ts_field != "timestamp":
        df = df.rename(columns={ts_field: "timestamp"})

    # Normalise title column name
    if "headline" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"headline": "title"})

    log.info(f"[FETCH] news → {len(df)} rows.")
    return df


def fetch_events(db, lookback_days: int = 730) -> pd.DataFrame:
    """Fetch macro event data from the events collection."""
    log.info("[FETCH] Loading events…")
    since = datetime.now(UTC) - timedelta(days=lookback_days)

    ts_candidates = ["timestamp", "date", "event_date", "published_at"]
    sample = db[EVENTS_COL].find_one()
    if sample is None:
        log.warning("[FETCH] Events collection is empty.")
        return pd.DataFrame()

    ts_field = next((f for f in ts_candidates if f in sample), ts_candidates[0])
    log.debug(f"[FETCH] Events timestamp field detected: '{ts_field}'")

    docs = list(db[EVENTS_COL].find(
        {ts_field: {"$gte": since}},
        {"_id": 0, ts_field: 1, "event_type": 1, "title": 1,
         "description": 1, "impact_score": 1},
    ))
    if not docs:
        log.warning("[FETCH] No event documents returned.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    if ts_field != "timestamp":
        df = df.rename(columns={ts_field: "timestamp"})

    log.info(f"[FETCH] events → {len(df)} rows.")
    return df


# ─────────────────────────────────────────────────────────────
# 2. CLEAN & NORMALISE EACH SOURCE
# ─────────────────────────────────────────────────────────────

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise price data and attach trading_date join key.
    Mirrors the conventions established in market_pipeline.py.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"]    = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df                 = df.dropna(subset=["timestamp"])
    df["trading_date"] = _normalize_ts_col(df["timestamp"])
    df                 = df.dropna(subset=["trading_date"])

    # Enforce numeric types
    for col in ["close", "returns", "rsi", "atr", "rolling_volatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # One record per trading day (keep last if duplicates exist)
    df = df.sort_values("timestamp").drop_duplicates(subset=["trading_date"], keep="last")
    df = df[["trading_date", "close", "returns", "rsi", "atr", "rolling_volatility"]]

    log.info(f"[CLEAN] prices → {len(df)} rows after cleaning.")
    return df.sort_values("trading_date").reset_index(drop=True)


def clean_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise flows data and attach trading_date join key.
    Mirrors the conventions established in flows_pipeline.py.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"]    = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df                 = df.dropna(subset=["timestamp"])
    df["trading_date"] = _normalize_ts_col(df["timestamp"])
    df                 = df.dropna(subset=["trading_date"])

    for col in ["total_flow", "flow_3d", "flow_5d", "flow_momentum", "flow_zscore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "flow_anomaly" in df.columns:
        df["flow_anomaly"] = df["flow_anomaly"].fillna(False).astype(bool)

    df = df.sort_values("timestamp").drop_duplicates(subset=["trading_date"], keep="last")
    flow_cols = ["trading_date", "total_flow", "flow_3d", "flow_5d",
                 "flow_momentum", "flow_zscore", "flow_anomaly"]
    df = df[[c for c in flow_cols if c in df.columns]]

    log.info(f"[CLEAN] flows → {len(df)} rows after cleaning.")
    return df.sort_values("trading_date").reset_index(drop=True)


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate news on (title + timestamp) and attach trading_date join key.
    Deduplication rule: keep the row with the latest raw timestamp.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("").str.strip()

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    # ── Deduplication: (title + published_at) ───────────────
    # Keep the latest version when the same article appears multiple times
    df = df.sort_values("timestamp", ascending=False)
    df = df.drop_duplicates(subset=["title", "timestamp"], keep="first")

    df["trading_date"] = _normalize_ts_col(df["timestamp"])
    df = df.dropna(subset=["trading_date"])

    log.info(f"[CLEAN] news → {len(df)} rows after deduplication.")
    return df.reset_index(drop=True)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate events on (event_type + timestamp) and attach trading_date.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    if "event_type" not in df.columns:
        df["event_type"] = ""
    df["event_type"] = df["event_type"].fillna("").str.strip().str.lower()

    if "impact_score" in df.columns:
        df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce")
    else:
        df["impact_score"] = np.nan

    # Combine title + description into a single searchable text field
    title_col = df.get("title", pd.Series("", index=df.index)).fillna("")
    desc_col  = df.get("description", pd.Series("", index=df.index)).fillna("")
    df["_text"] = (title_col + " " + desc_col).str.lower()

    # ── Deduplication: (event_type + timestamp) ─────────────
    df = df.sort_values("timestamp", ascending=False)
    df = df.drop_duplicates(subset=["event_type", "timestamp"], keep="first")

    df["trading_date"] = _normalize_ts_col(df["timestamp"])
    df = df.dropna(subset=["trading_date"])

    log.info(f"[CLEAN] events → {len(df)} rows after deduplication.")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. PROCESS NEWS → daily sentiment aggregates
# ─────────────────────────────────────────────────────────────

def process_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-article news sentiment into one row per trading_date.

    Produces:
        news_count           – number of articles on that day
        avg_sentiment        – mean sentiment score
        max_sentiment        – most positive article score
        min_sentiment        – most negative article score
        sentiment_volatility – std dev of sentiment (NaN if <2 articles)
    """
    if df.empty:
        log.warning("[NEWS] Empty DataFrame — returning empty news aggregates.")
        return pd.DataFrame(columns=[
            "trading_date", "news_count", "avg_sentiment",
            "max_sentiment", "min_sentiment", "sentiment_volatility",
        ])

    agg = (
        df.groupby("trading_date")["sentiment_score"]
        .agg(
            news_count="count",
            avg_sentiment="mean",
            max_sentiment="max",
            min_sentiment="min",
            sentiment_volatility="std",
        )
        .reset_index()
    )

    # Round for storage efficiency
    for col in ["avg_sentiment", "max_sentiment", "min_sentiment", "sentiment_volatility"]:
        agg[col] = agg[col].round(6)

    log.info(f"[NEWS] Aggregated → {len(agg)} trading-date rows.")
    return agg


# ─────────────────────────────────────────────────────────────
# 4. PROCESS EVENTS → daily category flags + counts
# ─────────────────────────────────────────────────────────────

def _classify_event(row: pd.Series) -> str:
    """
    Return the EventOracle category for a single event row.
    Checks event_type field first, then falls back to full-text search.
    Returns 'other' if no category matches.
    """
    event_type = str(row.get("event_type", "")).lower()
    text       = str(row.get("_text", "")).lower()
    combined   = event_type + " " + text

    for category, keywords in EVENT_CATEGORY_MAP.items():
        if any(kw in combined for kw in keywords):
            return category
    return "other"


def process_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify events into 8 categories and aggregate per trading_date.

    Produces:
        event_count          – total events on that day
        event_impact_score   – mean impact_score (NaN if unavailable)
        inflation_flag       – 1 if any inflation event on that day
        growth_flag
        central_bank_flag
        liquidity_flag
        regulation_flag
        geopolitical_flag
        oil_flag
        other_flag
    """
    category_cols = list(EVENT_CATEGORY_MAP.keys()) + ["other"]

    empty_cols = (
        ["trading_date", "event_count", "event_impact_score"]
        + [f"{c}_flag" for c in category_cols]
    )
    if df.empty:
        log.warning("[EVENTS] Empty DataFrame — returning empty event aggregates.")
        return pd.DataFrame(columns=empty_cols)

    df = df.copy()
    df["category"] = df.apply(_classify_event, axis=1)

    # Count & impact per day
    base_agg = (
        df.groupby("trading_date")
        .agg(
            event_count=("category", "count"),
            event_impact_score=("impact_score", "mean"),
        )
        .reset_index()
    )
    base_agg["event_impact_score"] = base_agg["event_impact_score"].round(4)

    # One-hot category flags (1 = at least one event in that category that day)
    cat_dummies = pd.get_dummies(df[["trading_date", "category"]], columns=["category"])
    cat_agg = (
        cat_dummies.groupby("trading_date")
        .max()
        .reset_index()
    )
    # Rename to *_flag and ensure all 8 categories exist even if absent in data
    rename_map = {f"category_{c}": f"{c}_flag" for c in category_cols}
    cat_agg = cat_agg.rename(columns=rename_map)
    for c in category_cols:
        flag = f"{c}_flag"
        if flag not in cat_agg.columns:
            cat_agg[flag] = 0
    flag_cols = [f"{c}_flag" for c in category_cols]
    cat_agg[flag_cols] = cat_agg[flag_cols].fillna(0).astype(int)

    result = base_agg.merge(cat_agg[["trading_date"] + flag_cols], on="trading_date", how="left")
    log.info(f"[EVENTS] Aggregated → {len(result)} trading-date rows.")
    return result


# ─────────────────────────────────────────────────────────────
# 5. FINAL MERGE
# ─────────────────────────────────────────────────────────────

def merge_all(
    prices_df:  pd.DataFrame,
    flows_df:   pd.DataFrame,
    news_agg:   pd.DataFrame,
    events_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join everything onto the price spine using trading_date.

    Merge order:
        prices (spine) → flows → news aggregates → event aggregates

    Missing value strategy:
        - Price features    : forward-fill (market closed / data gap)
        - Flow features     : fill 0 (no flow activity)
        - Sentiment         : fill 0 (neutral)
        - Event features    : fill 0 (no event)
        - flow_anomaly      : fill False

    Finally computes the target variable:
        target = next day's return (returns.shift(-1))
    """
    if prices_df.empty:
        log.error("[MERGE] Price spine is empty — cannot produce features.")
        return pd.DataFrame()

    # ── 1. Spine: prices ────────────────────────────────────
    df = prices_df.copy().sort_values("trading_date")

    # ── 2. Flows ────────────────────────────────────────────
    if not flows_df.empty:
        df = df.merge(flows_df, on="trading_date", how="left")
    else:
        log.warning("[MERGE] Flows DataFrame empty — flow columns will be 0.")
        for col in ["total_flow", "flow_3d", "flow_5d",
                    "flow_momentum", "flow_zscore", "flow_anomaly"]:
            df[col] = np.nan

    # ── 3. News aggregates ──────────────────────────────────
    if not news_agg.empty:
        df = df.merge(news_agg, on="trading_date", how="left")
    else:
        log.warning("[MERGE] News aggregates empty — sentiment columns will be 0.")
        for col in ["news_count", "avg_sentiment", "max_sentiment",
                    "min_sentiment", "sentiment_volatility"]:
            df[col] = np.nan

    # ── 4. Event aggregates ─────────────────────────────────
    if not events_agg.empty:
        df = df.merge(events_agg, on="trading_date", how="left")
    else:
        log.warning("[MERGE] Event aggregates empty — event columns will be 0.")
        category_cols = list(EVENT_CATEGORY_MAP.keys()) + ["other"]
        df["event_count"]        = np.nan
        df["event_impact_score"] = np.nan
        for c in category_cols:
            df[f"{c}_flag"] = 0

    # ── 5. Missing value strategy ───────────────────────────
    price_feat_cols = ["close", "returns", "rsi", "atr", "rolling_volatility"]
    flow_feat_cols  = ["total_flow", "flow_3d", "flow_5d",
                       "flow_momentum", "flow_zscore"]
    sentiment_cols  = ["news_count", "avg_sentiment", "max_sentiment",
                       "min_sentiment", "sentiment_volatility"]
    event_num_cols  = ["event_count", "event_impact_score"]
    category_cols   = list(EVENT_CATEGORY_MAP.keys()) + ["other"]
    flag_cols       = [f"{c}_flag" for c in category_cols]

    # Forward-fill price features (handles market holidays / missing days)
    df = df.sort_values("trading_date")
    for col in price_feat_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Fill flow NaNs with 0 (no institutional activity data = treat as zero)
    for col in flow_feat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # flow_anomaly: False where missing
    if "flow_anomaly" in df.columns:
        df["flow_anomaly"] = df["flow_anomaly"].fillna(False).astype(bool)

    # Sentiment: 0 = neutral
    for col in sentiment_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Events: 0 = no event
    for col in event_num_cols + flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── 6. Target variable: next-day return ─────────────────
    df["target"] = df["returns"].shift(-1)
    # Last row will have NaN target (no future data) — intentional

    # ── 7. Final column ordering ────────────────────────────
    desired_columns = [
        "trading_date",
        # Price features
        "close", "returns", "rsi", "atr", "rolling_volatility",
        # Flow features
        "total_flow", "flow_3d", "flow_5d",
        "flow_momentum", "flow_zscore", "flow_anomaly",
        # News / sentiment features
        "news_count", "avg_sentiment", "max_sentiment",
        "min_sentiment", "sentiment_volatility",
        # Event features
        "event_count", "event_impact_score",
        "inflation_flag", "growth_flag", "central_bank_flag",
        "liquidity_flag", "regulation_flag",
        "geopolitical_flag", "oil_flag", "other_flag",
        # Target
        "target",
    ]
    present = [c for c in desired_columns if c in df.columns]
    df = df[present].reset_index(drop=True)

    log.info(
        f"[MERGE] Unified features: {len(df)} rows × {len(df.columns)} cols | "
        f"Date range: {df['trading_date'].min()} → {df['trading_date'].max()}"
    )
    return df


# ─────────────────────────────────────────────────────────────
# 6. DATABASE OUTPUT
# ─────────────────────────────────────────────────────────────

def upload_to_mongo(df: pd.DataFrame, db, dry_run: bool = False) -> int:
    """
    Upsert unified feature records into MongoDB Atlas.
    trading_date is stored as a Python date-string (ISO format) for readability
    and as a proper datetime (UTC midnight) for time-series queries.

    Returns count of newly inserted documents.
    """
    if df.empty:
        log.warning("[MONGO] Nothing to upload — DataFrame is empty.")
        return 0

    records = df.to_dict("records")

    for rec in records:
        td = rec.get("trading_date")

        # ── Dual storage: human-readable string + queryable datetime ──
        if isinstance(td, date):
            rec["trading_date_str"] = td.isoformat()          # "2024-05-16"
            rec["trading_date"]     = datetime(                # UTC midnight
                td.year, td.month, td.day,
                tzinfo=timezone.utc,
            )
        elif isinstance(td, str):
            rec["trading_date_str"] = td
            try:
                parsed = datetime.strptime(td, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                rec["trading_date"] = parsed
            except ValueError:
                pass

        # ── Serialise pandas / numpy types for BSON ──────────────────
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v) if np.isfinite(v) else None

    if dry_run:
        log.info(
            f"[MONGO] DRY RUN — would upsert {len(records)} records "
            f"into '{FEATURES_COL}'."
        )
        return 0

    collection = db[FEATURES_COL]
    ops = [
        UpdateOne(
            {"trading_date": rec["trading_date"]},
            {"$set": rec},          # $set (not $setOnInsert) so features refresh daily
            upsert=True,
        )
        for rec in records
    ]

    total_upserted = 0
    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count + result.modified_count
        except BulkWriteError as bwe:
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"[MONGO] BulkWriteError (expected on reruns): {bwe.details}")

    log.info(
        f"[MONGO] ✅ Upserted/updated {total_upserted} records "
        f"into '{FEATURES_COL}'."
    )
    return total_upserted


def export_csv(df: pd.DataFrame, path: str = "features.csv") -> None:
    """Save the unified features DataFrame to a CSV file."""
    df_out = df.copy()
    # trading_date may be date objects — convert to string for CSV readability
    df_out["trading_date"] = df_out["trading_date"].astype(str)
    df_out.to_csv(path, index=False)
    log.info(f"[CSV] Features exported → {path} ({len(df_out)} rows).")


# ─────────────────────────────────────────────────────────────
# 7. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    lookback_days: int = 730,
    dry_run:       bool = False,
    save_csv:      bool = False,
    csv_path:      str  = "features.csv",
) -> None:
    """
    Full pipeline orchestrator:
        fetch → clean → process → merge → upload (→ optional CSV)

    Args:
        lookback_days : Days of history to fetch from each collection.
        dry_run       : If True, skip all MongoDB writes.
        save_csv      : If True, also export features.csv.
        csv_path      : Path for the optional CSV export.
    """
    log.info("=" * 65)
    log.info("EventOracle – Data Cleaning & Feature Unification Pipeline")
    log.info(f"Lookback: {lookback_days}d | DryRun: {dry_run} | CSV: {save_csv}")
    log.info("=" * 65)

    # ── Connect to MongoDB ────────────────────────────────────
    try:
        client, db = get_mongo_db()
        ensure_indexes(db)
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as exc:
        log.error(f"❌ MongoDB connection failed: {exc}")
        return

    # ── Stage 1: Fetch ───────────────────────────────────────
    log.info("\n── Stage 1: Fetching from MongoDB ──")
    raw_prices = fetch_prices(db, lookback_days=lookback_days)
    raw_flows  = fetch_flows(db,  lookback_days=lookback_days)
    raw_news   = fetch_news(db,   lookback_days=lookback_days)
    raw_events = fetch_events(db, lookback_days=lookback_days)

    if raw_prices.empty:
        log.error("❌ Pipeline aborted — no price data available (required spine).")
        client.close()
        return

    # ── Stage 2: Clean & normalise each source ───────────────
    log.info("\n── Stage 2: Cleaning & normalising ──")
    clean_prices_df = clean_prices(raw_prices)
    clean_flows_df  = clean_flows(raw_flows)
    clean_news_df   = clean_news(raw_news)
    clean_events_df = clean_events(raw_events)

    # ── Stage 3: Process news → daily aggregates ─────────────
    log.info("\n── Stage 3: Processing news ──")
    news_agg = process_news(clean_news_df)

    # ── Stage 4: Process events → category flags ─────────────
    log.info("\n── Stage 4: Processing events ──")
    events_agg = process_events(clean_events_df)

    # ── Stage 5: Merge all on trading_date ───────────────────
    log.info("\n── Stage 5: Merging all sources on trading_date ──")
    features_df = merge_all(
        prices_df  = clean_prices_df,
        flows_df   = clean_flows_df,
        news_agg   = news_agg,
        events_agg = events_agg,
    )

    if features_df.empty:
        log.error("❌ Pipeline aborted — merge produced empty DataFrame.")
        client.close()
        return

    # ── Stage 6: Upload to MongoDB ───────────────────────────
    log.info("\n── Stage 6: Uploading to MongoDB ──")
    inserted = upload_to_mongo(features_df, db, dry_run=dry_run)

    # ── Stage 7 (optional): CSV export ───────────────────────
    if save_csv:
        log.info("\n── Stage 7: Exporting CSV ──")
        export_csv(features_df, path=csv_path)

    # ── Summary ──────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("  EventOracle – Feature Unification Pipeline Summary")
    log.info("=" * 65)
    log.info(f"  Price rows (spine)     : {len(clean_prices_df)}")
    log.info(f"  Flow rows              : {len(clean_flows_df)}")
    log.info(f"  News rows (raw)        : {len(clean_news_df)}")
    log.info(f"  News agg rows          : {len(news_agg)}")
    log.info(f"  Event rows (raw)       : {len(clean_events_df)}")
    log.info(f"  Event agg rows         : {len(events_agg)}")
    log.info(f"  Final feature rows     : {len(features_df)}")
    log.info(f"  Final feature cols     : {len(features_df.columns)}")
    log.info(f"  Columns                : {list(features_df.columns)}")
    log.info(f"  Date range             : "
             f"{features_df['trading_date'].min()} → "
             f"{features_df['trading_date'].max()}")
    if "target" in features_df.columns:
        non_null_target = features_df["target"].notna().sum()
        log.info(f"  Target (next-day ret)  : {non_null_target} non-null rows")
    log.info(f"  Records upserted       : {inserted}")
    if save_csv:
        log.info(f"  CSV exported to        : {csv_path}")
    log.info("=" * 65)

    client.close()
    log.info("✅ Feature unification pipeline complete.")


# ─────────────────────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle – Data Cleaning & Feature Unification Pipeline"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=730,
        help="Days of historical data to load from each collection (default: 730).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Process all stages but skip MongoDB write.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Also export unified features as features.csv.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="features.csv",
        help="Path for the CSV export (default: features.csv).",
    )

    args = parser.parse_args()
    run_pipeline(
        lookback_days = args.lookback,
        dry_run       = args.dry_run,
        save_csv      = args.csv,
        csv_path      = args.csv_path,
    )