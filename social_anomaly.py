"""
social_anomaly.py
EventOracle – Social Proxy & Anomaly Detection Pipeline (Person 8)
Dependencies: P2 (News) + P3 (Market) outputs in MongoDB features collection.

Reads from: eventoracle.features  (one doc per asset per day)
Writes to:  eventoracle.social_anomaly

Tasks:
  1. Social Proxy Features  (news-based sentiment signals)
  2. Cross-domain Anomaly Flags
  3. Isolation Forest Anomaly Detection
  4. Bulk upsert to MongoDB social_anomaly collection

Usage:
    python social_anomaly.py                          # All assets, all dates
    python social_anomaly.py --asset NIFTY            # Single asset
    python social_anomaly.py --start 2024-01-01       # From date
    python social_anomaly.py --asset NIFTY --dry-run  # Preview, no write

Scheduling (cron – daily at 7:00 AM IST / 1:30 AM UTC):
    30 1 * * * /usr/bin/python3 /path/to/social_anomaly.py >> /var/log/eventoracle.log 2>&1
"""

import os
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

MONGO_URI         = os.getenv("MONGO_URI")
DB_NAME           = "eventoracle"
SOURCE_COLLECTION = "features"
OUT_COLLECTION    = "social_anomaly"
IST               = timezone(timedelta(hours=5, minutes=30))
BATCH_SIZE        = 500
FFILL_LIMIT       = 3

IF_N_ESTIMATORS   = 100
IF_CONTAMINATION  = 0.05
IF_RANDOM_STATE   = 42

IF_FEATURES = [
    "returns",
    "rsi",
    "rolling_volatility",
    "volume_zscore",
    "avg_sentiment",
    "news_velocity",
    "flow_zscore",
]

KNOWN_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.social_anomaly")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def get_mongo_client() -> MongoClient:
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set in environment variables.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client


def load_features(
    client: MongoClient,
    asset: Optional[str] = None,
    start: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load feature rows from MongoDB with optional asset/date filters."""
    col = client[DB_NAME][SOURCE_COLLECTION]

    query: dict = {}
    if asset:
        query["asset"] = asset
    if start:
        query["timestamp"] = {"$gte": start}

    projection = {
        "_id": 0,
        "timestamp": 1,
        "asset": 1,
        # Price features
        "returns": 1,
        "rsi": 1,
        "rolling_volatility": 1,
        "volume_zscore": 1,
        # News features
        "avg_sentiment": 1,
        "news_volume": 1,
        "news_velocity": 1,
        "news_spike_flag": 1,
        # Flow features
        "flow_zscore": 1,
    }

    cursor = col.find(query, projection)
    df = pd.DataFrame(list(cursor))

    if df.empty:
        log.warning("No documents returned from features collection.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    log.info(f"Loaded {len(df)} rows from '{SOURCE_COLLECTION}' "
             f"({df['asset'].nunique()} assets).")
    return df


# ─────────────────────────────────────────────
# 2. CLEANING & IMPUTATION
# ─────────────────────────────────────────────

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset forward-fill (max 3 periods) then drop remaining nulls
    in columns required for downstream computation.
    """
    numeric_cols = [
        "returns", "rsi", "rolling_volatility", "volume_zscore",
        "avg_sentiment", "news_volume", "news_velocity", "news_spike_flag",
        "flow_zscore",
    ]

    # Ensure all expected columns exist (fill with NaN if absent)
    for col in numeric_cols:
        if col not in df.columns:
            log.warning(f"Column '{col}' missing from features – filling with NaN.")
            df[col] = np.nan

    df = df.sort_values(["asset", "timestamp"])

    # Per-asset ffill
    df[numeric_cols] = (
        df.groupby("asset", group_keys=False)[numeric_cols]
        .apply(lambda g: g.ffill(limit=FFILL_LIMIT))
    )

    before = len(df)
    # Only drop if the Isolation Forest features are all null
    df = df.dropna(subset=["returns", "avg_sentiment"])
    after = len(df)
    if before != after:
        log.info(f"Dropped {before - after} rows with unresolvable nulls.")

    df = df.drop_duplicates(subset=["timestamp", "asset"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. SOCIAL PROXY FEATURES
# ─────────────────────────────────────────────

def _zscore_series(s: pd.Series) -> pd.Series:
    """Rolling z-score over the entire series (uses mean/std of available data)."""
    mean = s.expanding(min_periods=5).mean()
    std  = s.expanding(min_periods=5).std().replace(0, np.nan)
    return (s - mean) / std


def compute_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset computation of social proxy signals:
      - social_sentiment_score
      - social_buzz_intensity
      - sentiment_momentum   (avg_sentiment - 3-day lag)
      - sentiment_volatility (5-day rolling std of avg_sentiment)
    """
    result_parts = []

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").copy()

        z_velocity = _zscore_series(grp["news_velocity"].fillna(0))
        z_volume   = _zscore_series(grp["news_volume"].fillna(0))

        grp["social_sentiment_score"] = (
            0.5 * grp["avg_sentiment"].fillna(0)
            + 0.3 * z_velocity.fillna(0)
            + 0.2 * z_volume.fillna(0)
        ).round(6)

        grp["social_buzz_intensity"] = z_volume.round(6)

        grp["sentiment_momentum"] = (
            grp["avg_sentiment"] - grp["avg_sentiment"].shift(3)
        ).round(6)

        grp["sentiment_volatility"] = (
            grp["avg_sentiment"].rolling(5, min_periods=2).std()
        ).round(6)

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ─────────────────────────────────────────────
# 4. CROSS-DOMAIN ANOMALY FLAGS
# ─────────────────────────────────────────────

def compute_cross_domain_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset rolling signals:
      - flow_anomaly_flag   : |flow_zscore| > 2
      - price_anomaly_flag  : |returns| > 2σ (rolling 10)
      - market_stress_score : sum of the three binary flags
      - composite_anomaly_flag : market_stress_score >= 2
    """
    result_parts = []

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").copy()

        flow_z = grp["flow_zscore"].fillna(0)
        grp["flow_anomaly_flag"] = (flow_z.abs() > 2).astype(int)

        ret = grp["returns"].fillna(0)
        roll_std = ret.rolling(10, min_periods=3).std().replace(0, np.nan)
        grp["price_anomaly_flag"] = (ret.abs() > 2 * roll_std).astype(int)

        news_spike = grp["news_spike_flag"].fillna(0).astype(int)

        grp["market_stress_score"] = (
            news_spike
            + grp["flow_anomaly_flag"]
            + grp["price_anomaly_flag"]
        )

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ─────────────────────────────────────────────
# 5. ISOLATION FOREST ANOMALY DETECTION
# ─────────────────────────────────────────────

def compute_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Isolation Forest per asset on IF_FEATURES.
    Outputs:
      anomaly_score  : raw decision function (higher = more normal)
      anomaly_flag   : 1 if anomaly, 0 if normal
    """
    result_parts = []

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").copy()
        grp["anomaly_score"] = np.nan
        grp["anomaly_flag"]  = 0

        available_features = [f for f in IF_FEATURES if f in grp.columns]
        feature_df = grp[available_features].copy()

        # Rows where all IF features are present
        valid_mask = feature_df.notna().all(axis=1)
        n_valid = valid_mask.sum()

        if n_valid < 20:
            log.warning(
                f"[{asset}] Only {n_valid} complete rows for IF – skipping anomaly detection."
            )
            result_parts.append(grp)
            continue

        X = feature_df.loc[valid_mask].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )
        iso.fit(X_scaled)

        scores = iso.decision_function(X_scaled)   # higher = more normal
        preds  = iso.predict(X_scaled)              # -1 = anomaly, 1 = normal

        grp.loc[valid_mask, "anomaly_score"] = np.round(scores, 6)
        grp.loc[valid_mask, "anomaly_flag"]  = (preds == -1).astype(int)

        n_anomalies = (preds == -1).sum()
        log.info(f"[{asset}] Isolation Forest: {n_anomalies}/{n_valid} anomalies detected.")

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ─────────────────────────────────────────────
# 6. COMPOSITE FLAG & FINAL SCHEMA
# ─────────────────────────────────────────────

OUTPUT_FIELDS = [
    "timestamp", "asset",
    "social_sentiment_score", "social_buzz_intensity",
    "sentiment_momentum", "sentiment_volatility",
    "anomaly_score", "anomaly_flag",
    "market_stress_score", "composite_anomaly_flag",
]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble final output schema and compute composite_anomaly_flag."""
    df["composite_anomaly_flag"] = (df["market_stress_score"] >= 2).astype(int)

    # Ensure all output fields exist
    for field in OUTPUT_FIELDS:
        if field not in df.columns:
            df[field] = np.nan

    out = df[OUTPUT_FIELDS].copy()
    out = out.drop_duplicates(subset=["timestamp", "asset"])
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────
# 7. DATABASE STORAGE
# ─────────────────────────────────────────────

def get_output_collection(client: MongoClient):
    col = client[DB_NAME][OUT_COLLECTION]
    col.create_index(
        [("timestamp", 1), ("asset", 1)],
        unique=True,
        background=True,
        name="timestamp_asset_unique",
    )
    return col


def upload_to_mongo(df: pd.DataFrame, collection, dry_run: bool = False) -> int:
    """Bulk upsert records into MongoDB. Returns count of upserted docs."""
    if df.empty:
        log.warning("Nothing to upload – output dataframe is empty.")
        return 0

    records = df.to_dict("records")

    # Coerce types for MongoDB
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v) if not np.isnan(v) else None

    ops = [
        UpdateOne(
            {"timestamp": rec["timestamp"], "asset": rec["asset"]},
            {"$set": rec},
            upsert=True,
        )
        for rec in records
    ]

    if dry_run:
        log.info(f"[DRY RUN] Would upsert {len(ops)} records into '{OUT_COLLECTION}'.")
        return 0

    total_upserted = 0
    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count + result.modified_count
        except BulkWriteError as bwe:
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"BulkWriteError details: {bwe.details}")

    log.info(f"✅ Upserted/updated {total_upserted} records into '{OUT_COLLECTION}'.")
    return total_upserted


# ─────────────────────────────────────────────
# 8. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(
    asset: Optional[str] = None,
    start: Optional[str] = None,
    dry_run: bool = False,
):
    log.info("=" * 60)
    log.info("EventOracle – Social & Anomaly Pipeline Starting")
    log.info(f"  Asset   : {asset or 'ALL'}")
    log.info(f"  Start   : {start or 'ALL TIME'}")
    log.info(f"  Dry Run : {dry_run}")
    log.info("=" * 60)

    # Parse start date
    start_dt: Optional[datetime] = None
    if start:
        try:
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        except ValueError:
            log.error(f"Invalid --start date format '{start}'. Use YYYY-MM-DD.")
            return

    try:
        client = get_mongo_client()
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as e:
        log.error(f"❌ MongoDB connection failed: {e}")
        return

    # ── Step 1: Load ──
    try:
        raw_df = load_features(client, asset=asset, start=start_dt)
    except Exception as e:
        log.error(f"Failed to load features: {e}", exc_info=True)
        return

    if raw_df.empty:
        log.warning("No data to process. Exiting.")
        return

    # ── Step 2: Clean ──
    log.info("Cleaning and imputing data...")
    df = clean_and_impute(raw_df)
    if df.empty:
        log.warning("All rows dropped after cleaning. Exiting.")
        return

    # ── Step 3: Social Features ──
    log.info("Computing social proxy features...")
    df = compute_social_features(df)

    # ── Step 4: Cross-domain Flags ──
    log.info("Computing cross-domain anomaly flags...")
    df = compute_cross_domain_flags(df)

    # ── Step 5: Isolation Forest ──
    log.info("Running Isolation Forest anomaly detection...")
    df = compute_isolation_forest(df)

    # ── Step 6: Final Schema ──
    output_df = build_output(df)
    log.info(f"Output shape: {output_df.shape}")

    if dry_run:
        log.info("\n── DRY RUN PREVIEW (first 10 rows) ──")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            log.info("\n" + output_df.head(10).to_string(index=False))

    # ── Step 7: Upload ──
    try:
        col = get_output_collection(client)
        upload_to_mongo(output_df, col, dry_run=dry_run)
    except Exception as e:
        log.error(f"Upload failed: {e}", exc_info=True)
        return

    log.info("=" * 60)
    log.info("Pipeline Complete.")
    log.info("=" * 60)


# ─────────────────────────────────────────────
# 9. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle Social Proxy & Anomaly Detection Pipeline (P8)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        choices=KNOWN_ASSETS,
        help="Run for a single asset (default: all assets)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Process only rows on or after this date (UTC)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute features but do NOT write to MongoDB",
    )
    args = parser.parse_args()

    run_pipeline(
        asset=args.asset,
        start=args.start,
        dry_run=args.dry_run,
    )