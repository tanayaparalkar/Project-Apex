"""
social_anomaly.py
EventOracle – Social Sentiment & Anomaly Detection Pipeline (Person 8)

Reads from:
  - "news"   collection  → sentiment proxy for #Nifty / market keywords
  - "flows"  collection  → institutional flow features (Person 6)
  - "prices" collection  → market OHLCV + technical features (Person 3)

Writes to:
  - "features" collection → combined alt-data feature rows with anomaly flags

Signal logic:
  anomaly_flag == 1 AND sentiment_score <  0  →  STRONG SELL
  anomaly_flag == 1 AND sentiment_score >  0  →  STRONG BUY
  otherwise                                   →  HOLD

Usage:
    python social_anomaly.py                    # Run full pipeline (last 3 days)
    python social_anomaly.py --lookback 7       # Extend lookback window
    python social_anomaly.py --dry-run          # Skip MongoDB write

Scheduling (cron – daily at 8:00 AM IST / 2:30 AM UTC, after upstream pipelines):
    30 2 * * * /usr/bin/python3 /path/to/social_anomaly.py >> /var/log/eventoracle_social.log 2>&1

GitHub Actions: schedule: cron: '30 2 * * *'

Dependencies:
    pip install pymongo python-dotenv pandas numpy scikit-learn pytz
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.social_anomaly")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI    = os.getenv("MONGO_URI")
DB_NAME      = "eventoracle"
NEWS_COL     = "news"
FLOWS_COL    = "flows"
PRICES_COL   = "prices"
FEATURES_COL = "features"

IST        = pytz.timezone("Asia/Kolkata")
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))

# News keywords that act as a Twitter #Nifty sentiment proxy
SENTIMENT_KEYWORDS = ["nifty", "banknifty", "bank nifty", "market", "stocks", "sensex", "nse", "bse"]

# Primary asset for price/flow alignment
PRIMARY_ASSET = "NIFTY"

# IsolationForest
CONTAMINATION      = 0.05
RANDOM_STATE       = 42

# Rolling windows
WINDOW_3D          = 3
WINDOW_5D          = 5
ZSCORE_WINDOW      = 20

# Feature columns fed into IsolationForest
ANOMALY_FEATURES   = [
    "returns",
    "volatility",
    "volume_zscore",
    "total_flow",
    "flow_zscore",
    "sentiment_score",
    "sentiment_volatility",
]

BATCH_SIZE = 500


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(IST)


def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce any datetime to IST-aware. Assumes UTC if naive."""
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def _floor_to_day(dt: datetime) -> datetime:
    """Truncate IST-aware datetime to midnight of that day."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────

def fetch_news_sentiment(db, lookback_days: int = 3) -> pd.DataFrame:
    """
    Pull recent news documents matching market/Nifty keywords.
    Returns DataFrame with: timestamp, sentiment_score.
    Acts as a Twitter #Nifty sentiment proxy using the news collection.
    """
    log.info(f"[FETCH] News sentiment — last {lookback_days} day(s), keywords: {SENTIMENT_KEYWORDS}")
    cutoff = ist_now() - timedelta(days=lookback_days)

    try:
        col = db[NEWS_COL]
        keyword_regex = "|".join(SENTIMENT_KEYWORDS)
        cursor = col.find(
            {
                "timestamp": {"$gte": cutoff},
                "headline": {"$regex": keyword_regex, "$options": "i"},
                "sentiment_score": {"$exists": True},
            },
            {"_id": 0, "timestamp": 1, "sentiment_score": 1, "headline": 1},
        )
        docs = list(cursor)
    except Exception as exc:
        log.error(f"[FETCH] News query failed: {exc}")
        return pd.DataFrame()

    if not docs:
        log.warning("[FETCH] No matching news articles found in lookback window.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(to_ist)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["timestamp", "sentiment_score"])

    log.info(f"[FETCH] News: {len(df)} articles retrieved.")
    return df.reset_index(drop=True)


def fetch_flows(db, lookback_days: int = 30) -> pd.DataFrame:
    """
    Pull recent flow records from the flows collection.
    Returns DataFrame with: timestamp, total_flow, flow_zscore.
    """
    log.info(f"[FETCH] Flows — last {lookback_days} day(s).")
    cutoff = ist_now() - timedelta(days=lookback_days)

    try:
        col = db[FLOWS_COL]
        cursor = col.find(
            {"timestamp": {"$gte": cutoff}},
            {"_id": 0, "timestamp": 1, "total_flow": 1, "flow_zscore": 1},
        )
        docs = list(cursor)
    except Exception as exc:
        log.error(f"[FETCH] Flows query failed: {exc}")
        return pd.DataFrame()

    if not docs:
        log.warning("[FETCH] No flow records found in lookback window.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(to_ist)

    # Derive total_flow if missing (fii_net + dii_net may exist instead)
    if "total_flow" not in df.columns:
        for candidate in ["fii_net", "net_flow", "flow"]:
            if candidate in df.columns:
                df["total_flow"] = pd.to_numeric(df[candidate], errors="coerce")
                break
        else:
            df["total_flow"] = np.nan

    df["total_flow"]  = pd.to_numeric(df.get("total_flow",  np.nan), errors="coerce")
    df["flow_zscore"] = pd.to_numeric(df.get("flow_zscore", np.nan), errors="coerce")
    df = df.dropna(subset=["timestamp"])

    log.info(f"[FETCH] Flows: {len(df)} records retrieved.")
    return df[["timestamp", "total_flow", "flow_zscore"]].reset_index(drop=True)


def fetch_prices(db, lookback_days: int = 30, asset: str = PRIMARY_ASSET) -> pd.DataFrame:
    """
    Pull recent price records for the primary asset.
    Returns DataFrame with: timestamp, returns, rolling_volatility, volume_zscore.
    """
    log.info(f"[FETCH] Prices — last {lookback_days} day(s), asset: {asset}.")
    cutoff = ist_now() - timedelta(days=lookback_days)

    try:
        col = db[PRICES_COL]
        cursor = col.find(
            {"timestamp": {"$gte": cutoff}, "asset": asset},
            {
                "_id": 0,
                "timestamp": 1,
                "returns": 1,
                "rolling_volatility": 1,
                "volume_zscore": 1,
                "close": 1,
            },
        )
        docs = list(cursor)
    except Exception as exc:
        log.error(f"[FETCH] Prices query failed: {exc}")
        return pd.DataFrame()

    if not docs:
        log.warning(f"[FETCH] No price records for {asset} in lookback window.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(to_ist)

    for col_name in ["returns", "rolling_volatility", "volume_zscore"]:
        df[col_name] = pd.to_numeric(df.get(col_name, np.nan), errors="coerce")

    df = df.dropna(subset=["timestamp"])
    log.info(f"[FETCH] Prices: {len(df)} records retrieved.")
    return df[["timestamp", "returns", "rolling_volatility", "volume_zscore"]].reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────

def clean_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw news articles into daily sentiment buckets (IST calendar day).
    Output columns: date (IST day), raw_sentiment_scores (list).
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = df["timestamp"].apply(_floor_to_day)
    df = df.dropna(subset=["sentiment_score", "date"])
    log.info(f"[CLEAN] News: {len(df)} articles after cleaning.")
    return df.reset_index(drop=True)


def clean_flows_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize flow timestamps to IST day boundaries and drop duplicates."""
    if df.empty:
        return df

    df = df.copy()
    df["date"] = df["timestamp"].apply(_floor_to_day)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    log.info(f"[CLEAN] Flows: {len(df)} records after cleaning.")
    return df.reset_index(drop=True)


def clean_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize price timestamps to IST day boundaries and drop duplicates."""
    if df.empty:
        return df

    df = df.copy()
    df["date"] = df["timestamp"].apply(_floor_to_day)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    log.info(f"[CLEAN] Prices: {len(df)} records after cleaning.")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. ADD FEATURES
# ─────────────────────────────────────────────

def _compute_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-day sentiment features from the news DataFrame.

    Returns DataFrame with columns:
        date, avg_sentiment_score, sentiment_volatility, sentiment_trend
    """
    if news_df.empty:
        return pd.DataFrame(columns=["date", "avg_sentiment_score", "sentiment_volatility", "sentiment_trend"])

    agg = (
        news_df.groupby("date")["sentiment_score"]
        .agg(
            avg_sentiment_score="mean",
            sentiment_volatility="std",
        )
        .reset_index()
    )

    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0.0)

    # Sentiment trend: rolling mean change over WINDOW_3D
    agg = agg.sort_values("date")
    rolling_mean = agg["avg_sentiment_score"].rolling(WINDOW_3D, min_periods=1).mean()
    agg["sentiment_trend"] = rolling_mean.diff().fillna(0.0)

    agg = agg.rename(columns={"avg_sentiment_score": "sentiment_score"})
    agg["sentiment_score"]     = agg["sentiment_score"].round(6)
    agg["sentiment_volatility"] = agg["sentiment_volatility"].round(6)
    agg["sentiment_trend"]     = agg["sentiment_trend"].round(6)

    log.info(f"[FEATURES] Sentiment computed for {len(agg)} days.")
    return agg.reset_index(drop=True)


def add_features(
    news_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge sentiment, flow, and price features into a single DataFrame.

    Output columns:
        timestamp, returns, volatility, volume_zscore,
        total_flow, flow_zscore,
        sentiment_score, sentiment_volatility, sentiment_trend
    """
    log.info("[FEATURES] Building combined feature frame…")

    # ── Sentiment features ───────────────────────────────────
    sentiment_df = _compute_sentiment_features(news_df)

    # ── Prices baseline ──────────────────────────────────────
    if prices_df.empty:
        log.warning("[FEATURES] Prices DataFrame is empty — using sentiment-only frame.")
        base = sentiment_df.rename(columns={"date": "timestamp"})
        base["returns"]       = np.nan
        base["volatility"]    = np.nan
        base["volume_zscore"] = np.nan
    else:
        base = prices_df.rename(columns={
            "rolling_volatility": "volatility",
            "date": "timestamp",
        })[["timestamp", "returns", "volatility", "volume_zscore"]]

    # ── Merge flows ──────────────────────────────────────────
    if not flows_df.empty:
        flows_keyed = flows_df.rename(columns={"date": "timestamp"})[
            ["timestamp", "total_flow", "flow_zscore"]
        ]
        base = base.merge(flows_keyed, on="timestamp", how="left")
    else:
        log.warning("[FEATURES] Flows DataFrame is empty — filling with NaN.")
        base["total_flow"]  = np.nan
        base["flow_zscore"] = np.nan

    # ── Merge sentiment ──────────────────────────────────────
    if not sentiment_df.empty:
        sent_keyed = sentiment_df.rename(columns={"date": "timestamp"})[
            ["timestamp", "sentiment_score", "sentiment_volatility", "sentiment_trend"]
        ]
        base = base.merge(sent_keyed, on="timestamp", how="left")
    else:
        log.warning("[FEATURES] Sentiment DataFrame is empty — filling with 0.0.")
        base["sentiment_score"]      = 0.0
        base["sentiment_volatility"] = 0.0
        base["sentiment_trend"]      = 0.0

    # ── Fill remaining NaNs ──────────────────────────────────
    base["sentiment_score"]      = base["sentiment_score"].fillna(0.0)
    base["sentiment_volatility"] = base["sentiment_volatility"].fillna(0.0)
    base["sentiment_trend"]      = base["sentiment_trend"].fillna(0.0)

    # ── Rolling enrichment on merged frame ──────────────────
    base = base.sort_values("timestamp").reset_index(drop=True)

    # 5-day rolling flow z-score recompute if sparse
    if "flow_zscore" in base.columns:
        fz_null = base["flow_zscore"].isna().sum()
        if fz_null > 0 and "total_flow" in base.columns:
            roll_mean = base["total_flow"].rolling(ZSCORE_WINDOW, min_periods=1).mean()
            roll_std  = base["total_flow"].rolling(ZSCORE_WINDOW, min_periods=1).std().replace(0, np.nan)
            derived   = ((base["total_flow"] - roll_mean) / roll_std).round(6)
            base["flow_zscore"] = base["flow_zscore"].fillna(derived)

    base = base.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    log.info(f"[FEATURES] Combined frame shape: {base.shape}")
    log.info(f"[FEATURES] Columns: {list(base.columns)}")
    return base


# ─────────────────────────────────────────────
# 4. ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IsolationForest on the ANOMALY_FEATURES columns.

    Adds:
        anomaly_flag  (int)   : 1 = anomaly, 0 = normal
        anomaly_score (float) : raw decision_function output (higher = more normal)
        combined_signal (str) : STRONG BUY / STRONG SELL / HOLD
    """
    if df.empty:
        log.warning("[ANOMALY] Empty DataFrame — skipping anomaly detection.")
        df["anomaly_flag"]    = 0
        df["anomaly_score"]   = np.nan
        df["combined_signal"] = "HOLD"
        return df

    df = df.copy()

    # Select only available feature columns
    available = [f for f in ANOMALY_FEATURES if f in df.columns]
    missing   = [f for f in ANOMALY_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"[ANOMALY] Missing features (will be excluded): {missing}")

    if not available:
        log.error("[ANOMALY] No feature columns available — aborting anomaly detection.")
        df["anomaly_flag"]    = 0
        df["anomaly_score"]   = np.nan
        df["combined_signal"] = "HOLD"
        return df

    # Build feature matrix — fill residual NaNs with column median
    X = df[available].copy()
    for col in X.columns:
        median = X[col].median()
        X[col] = X[col].fillna(median if not np.isnan(median) else 0.0)

    if len(X) < 2:
        log.warning("[ANOMALY] Too few rows for IsolationForest — defaulting to HOLD.")
        df["anomaly_flag"]    = 0
        df["anomaly_score"]   = np.nan
        df["combined_signal"] = "HOLD"
        return df

    # Standardise
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit IsolationForest
    clf = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100,
    )
    clf.fit(X_scaled)

    raw_preds  = clf.predict(X_scaled)           # -1 = anomaly, 1 = normal
    raw_scores = clf.decision_function(X_scaled)  # higher = more normal

    df["anomaly_flag"]  = np.where(raw_preds == -1, 1, 0).astype(int)
    df["anomaly_score"] = raw_scores.round(6)

    n_anomalies = int(df["anomaly_flag"].sum())
    log.info(
        f"[ANOMALY] IsolationForest complete. "
        f"Anomalies: {n_anomalies}/{len(df)} "
        f"({n_anomalies / len(df) * 100:.1f}%)"
    )

    # ── Combined signal ──────────────────────────────────────
    def _signal(row) -> str:
        if row["anomaly_flag"] == 1 and row["sentiment_score"] < 0:
            return "STRONG SELL"
        elif row["anomaly_flag"] == 1 and row["sentiment_score"] > 0:
            return "STRONG BUY"
        return "HOLD"

    sentiment_col = "sentiment_score" if "sentiment_score" in df.columns else None
    if sentiment_col:
        df["combined_signal"] = df.apply(_signal, axis=1)
    else:
        df["combined_signal"] = np.where(df["anomaly_flag"] == 1, "ANOMALY", "HOLD")

    signal_dist = df["combined_signal"].value_counts().to_dict()
    log.info(f"[ANOMALY] Signal distribution: {signal_dist}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. DATABASE STORAGE
# ─────────────────────────────────────────────

def get_mongo_db():
    """Connect to MongoDB Atlas and return (client, db) tuple."""
    if not MONGO_URI:
        raise ValueError(
            "MONGO_URI is not set. Add it to your .env file: MONGO_URI=mongodb+srv://..."
        )
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    db = client[DB_NAME]
    return client, db


def ensure_indexes(db) -> None:
    """Create compound unique index on (timestamp, feature_source)."""
    col = db[FEATURES_COL]
    col.create_index(
        [("timestamp", 1), ("feature_source", 1)],
        unique=True,
        background=True,
        name="timestamp_source_unique",
    )
    log.info(f"[MONGO] Indexes ensured on '{FEATURES_COL}'.")


def upload_to_mongo(df: pd.DataFrame, db, dry_run: bool = False) -> int:
    """
    Upsert feature records into the 'features' MongoDB collection.
    Uses $setOnInsert to avoid overwriting existing records on reruns.
    Returns count of newly inserted documents.
    """
    if df.empty:
        log.warning("[MONGO] Nothing to upload — DataFrame is empty.")
        return 0

    OUTPUT_COLS = [
        "timestamp",
        "sentiment_score",
        "sentiment_volatility",
        "flow_zscore",
        "volume_zscore",
        "returns",
        "anomaly_flag",
        "anomaly_score",
        "combined_signal",
        "feature_source",
        # Bonus extras stored for downstream consumers
        "volatility",
        "total_flow",
        "sentiment_trend",
    ]

    df = df.copy()
    df["feature_source"] = "social_anomaly"

    # Keep only columns that exist in the frame
    cols_to_store = [c for c in OUTPUT_COLS if c in df.columns]
    records = df[cols_to_store].to_dict("records")

    for rec in records:
        # Serialize timestamps
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        elif isinstance(ts, datetime) and ts.tzinfo is None:
            rec["timestamp"] = ts.replace(tzinfo=IST_OFFSET)

        # Replace NaN → None for clean BSON
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)

    if dry_run:
        log.info(f"[MONGO] DRY RUN — would upsert {len(records)} records into '{FEATURES_COL}'.")
        return 0

    ops = [
        UpdateOne(
            {"timestamp": rec["timestamp"], "feature_source": rec["feature_source"]},
            {"$setOnInsert": rec},
            upsert=True,
        )
        for rec in records
    ]

    total_upserted = 0
    collection = db[FEATURES_COL]

    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count
        except BulkWriteError as bwe:
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"[MONGO] BulkWriteError (expected on reruns): {bwe.details}")

    log.info(f"[MONGO] ✅ Upserted {total_upserted} new records into '{FEATURES_COL}'.")
    return total_upserted


# ─────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(lookback_days: int = 3, dry_run: bool = False) -> None:
    """
    Full social anomaly pipeline:
        fetch → clean → add_features → detect_anomalies → upload
    """
    log.info("=" * 60)
    log.info("EventOracle – Social Sentiment & Anomaly Pipeline Starting")
    log.info(f"Lookback: {lookback_days}d | DryRun: {dry_run}")
    log.info("=" * 60)

    # ── Connect ─────────────────────────────────────────────
    try:
        client, db = get_mongo_db()
        ensure_indexes(db)
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as exc:
        log.error(f"❌ MongoDB connection failed: {exc}")
        return

    # Use a wider lookback for market data so rolling windows have signal
    market_lookback = max(lookback_days + ZSCORE_WINDOW, 30)

    try:
        # ── Stage 1: Fetch ───────────────────────────────────
        log.info("\n── Stage 1: Fetching data from upstream collections ──")
        raw_news   = fetch_news_sentiment(db, lookback_days=lookback_days)
        raw_flows  = fetch_flows(db,  lookback_days=market_lookback)
        raw_prices = fetch_prices(db, lookback_days=market_lookback, asset=PRIMARY_ASSET)

        all_empty = raw_news.empty and raw_flows.empty and raw_prices.empty
        if all_empty:
            log.error("❌ All upstream data sources returned empty. Pipeline aborted.")
            client.close()
            return

        # ── Stage 2: Clean ───────────────────────────────────
        log.info("\n── Stage 2: Cleaning data ──")
        clean_news_data   = clean_news_df(raw_news)
        clean_flows_data  = clean_flows_df(raw_flows)
        clean_prices_data = clean_prices_df(raw_prices)

        # ── Stage 3: Feature Engineering ─────────────────────
        log.info("\n── Stage 3: Engineering features ──")
        featured_df = add_features(
            news_df=clean_news_data,
            flows_df=clean_flows_data,
            prices_df=clean_prices_data,
        )

        if featured_df.empty:
            log.error("❌ Feature engineering returned empty DataFrame. Pipeline aborted.")
            client.close()
            return

        # ── Stage 4: Anomaly Detection ───────────────────────
        log.info("\n── Stage 4: Detecting anomalies ──")
        result_df = detect_anomalies(featured_df)

        # ── Stage 5: Upload ──────────────────────────────────
        log.info("\n── Stage 5: Uploading to MongoDB ──")
        inserted = upload_to_mongo(result_df, db, dry_run=dry_run)

        # ── Summary ──────────────────────────────────────────
        log.info("")
        log.info("=" * 60)
        log.info("  EventOracle – Social Anomaly Pipeline Summary")
        log.info("=" * 60)
        log.info(f"  Feature rows produced    : {len(result_df)}")
        log.info(f"  New records inserted      : {inserted}")
        log.info(f"  Anomalies detected        : {int(result_df['anomaly_flag'].sum())}")

        if "combined_signal" in result_df.columns:
            signal_counts = result_df["combined_signal"].value_counts().to_dict()
            for sig, cnt in signal_counts.items():
                log.info(f"  Signal [{sig:<12}]  : {cnt}")

        if not result_df.empty and "timestamp" in result_df.columns:
            ts_min = result_df["timestamp"].min()
            ts_max = result_df["timestamp"].max()
            if hasattr(ts_min, "date"):
                log.info(f"  Date range               : {ts_min.date()} → {ts_max.date()}")

        log.info("=" * 60)

    except Exception as exc:
        log.error(f"❌ Unexpected pipeline error: {exc}", exc_info=True)
    finally:
        client.close()
        log.info("✅ Social anomaly pipeline complete.")


# ─────────────────────────────────────────────
# 7. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle – Social Sentiment & Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="Days of news to use for sentiment window (default: 3). "
             "Market data uses a wider auto-extended window.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Process all stages but skip the MongoDB write.",
    )
    args = parser.parse_args()

    run_pipeline(
        lookback_days=args.lookback,
        dry_run=args.dry_run,
    )