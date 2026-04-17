"""
market_pipeline.py
EventOracle – Market Data Pipeline (Person 3)
Fetches 2yr OHLCV for 6 assets, engineers features, stores in MongoDB Atlas.

Usage:
    python market_pipeline.py                  # Run all assets
    python market_pipeline.py --asset NIFTY    # Run single asset
    python market_pipeline.py --asset BRENT    # Run single asset

Scheduling (cron example – runs daily at 6:30 AM IST / 1:00 AM UTC):
    0 1 * * * /usr/bin/python3 /path/to/market_pipeline.py >> /var/log/eventoracle.log 2>&1

GitHub Actions: add a workflow with schedule: cron: '0 1 * * *'
"""

import os
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME     = "eventoracle"
COLLECTION  = "prices"
PERIOD      = "2y"
INTERVAL    = "1d"
IST         = timezone(timedelta(hours=5, minutes=30))
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

ASSETS = {
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "NIFTYIT":    "^CNXIT",
    "NIFTYMETAL": "^CNXMETAL",
    "INRUSD":     "INR=X",
    "BRENT":      "BZ=F",
    "GOLD":       "GC=F",
    "SILVER":     "SI=F",
    "PLATINUM":   "PL=F",
    "BITCOIN":    "BTC-USD",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.pipeline")


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def fetch_data(asset_name: str, ticker: str) -> pd.DataFrame:
    """Fetch 2 years of daily OHLCV data from yfinance with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"[{asset_name}] Fetching data (attempt {attempt})...")
            df = yf.download(
                ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                raise ValueError(f"Empty dataframe returned for {ticker}")
            df["asset"] = asset_name
            log.info(f"[{asset_name}] Fetched {len(df)} rows.")
            return df
        except Exception as e:
            log.warning(f"[{asset_name}] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"[{asset_name}] All {MAX_RETRIES} attempts failed. Skipping.")
                return pd.DataFrame()


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Standardize columns, convert to IST, remove nulls/duplicates."""
    if df.empty:
        return df

    # yfinance MultiIndex columns when single ticker – flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Rename to schema
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"[{asset_name}] Missing columns after rename: {missing}")
        return pd.DataFrame()

    # Reset index → timestamp column
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "timestamp"})

    # Convert timestamp → IST-aware datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(IST)

    # Enforce types
    df["asset"]  = asset_name
    df["open"]   = pd.to_numeric(df["open"],   errors="coerce")
    df["high"]   = pd.to_numeric(df["high"],   errors="coerce")
    df["low"]    = pd.to_numeric(df["low"],    errors="coerce")
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    # Drop nulls and duplicates
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "asset"])
    after = len(df)

    if before != after:
        log.info(f"[{asset_name}] Dropped {before - after} null/duplicate rows.")

    df = df[["timestamp", "asset", "open", "high", "low", "close", "volume"]]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def add_features(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Add returns, RSI, ATR, rolling volatility, volume z-score."""
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["returns"]            = df["close"].pct_change()
    df["rsi"]                = _rsi(df["close"], 14)
    df["atr"]                = _atr(df, 14)
    df["rolling_volatility"] = df["returns"].rolling(10).std()

    vol_mean              = df["volume"].rolling(20).mean()
    vol_std               = df["volume"].rolling(20).std().replace(0, np.nan)
    df["volume_zscore"]   = (df["volume"] - vol_mean) / vol_std

    # Round floats for storage efficiency
    float_cols = ["open", "high", "low", "close", "returns",
                  "rsi", "atr", "rolling_volatility", "volume_zscore"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(6)

    log.info(f"[{asset_name}] Features engineered. Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. DATABASE STORAGE
# ─────────────────────────────────────────────

def get_mongo_collection():
    """Return the prices collection from MongoDB Atlas."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set in environment variables.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    # Ping to validate connection
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]
    # Compound unique index to prevent duplicates
    col.create_index(
        [("timestamp", 1), ("asset", 1)],
        unique=True,
        background=True,
        name="timestamp_asset_unique",
    )
    return col


def upload_to_mongo(df: pd.DataFrame, asset_name: str, collection) -> int:
    """Upsert records into MongoDB; returns count of inserted/modified docs."""
    if df.empty:
        log.warning(f"[{asset_name}] Nothing to upload.")
        return 0

    records = df.to_dict("records")

    # Convert pandas Timestamp → Python datetime for MongoDB
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        # Replace NaN with None
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None

    ops = [
        UpdateOne(
            {"timestamp": rec["timestamp"], "asset": rec["asset"]},
            {"$setOnInsert": rec},
            upsert=True,
        )
        for rec in records
    ]

    BATCH_SIZE = 500
    total_upserted = 0

    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count
        except BulkWriteError as bwe:
            # Duplicate key errors are expected on reruns – not fatal
            upserted = bwe.details.get("nUpserted", 0)
            total_upserted += upserted
            log.debug(f"[{asset_name}] BulkWriteError details: {bwe.details}")

    log.info(f"[{asset_name}] ✅ Upserted {total_upserted} new records into '{COLLECTION}'.")
    return total_upserted


# ─────────────────────────────────────────────
# 5. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(assets: dict = None):
    """Run the full pipeline for specified assets (default: all)."""
    if assets is None:
        assets = ASSETS

    log.info("=" * 55)
    log.info("EventOracle – Market Data Pipeline Starting")
    log.info(f"Assets: {list(assets.keys())}")
    log.info("=" * 55)

    try:
        collection = get_mongo_collection()
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as e:
        log.error(f"❌ MongoDB connection failed: {e}")
        return

    summary = {}

    for asset_name, ticker in assets.items():
        log.info(f"\n── Processing: {asset_name} ({ticker}) ──")
        try:
            raw_df      = fetch_data(asset_name, ticker)
            clean_df    = clean_data(raw_df, asset_name)
            featured_df = add_features(clean_df, asset_name)
            count       = upload_to_mongo(featured_df, asset_name, collection)
            summary[asset_name] = count
        except Exception as e:
            log.error(f"[{asset_name}] Pipeline error: {e}", exc_info=True)
            summary[asset_name] = "ERROR"

    log.info("\n" + "=" * 55)
    log.info("Pipeline Complete – Summary")
    log.info("=" * 55)
    for asset, count in summary.items():
        log.info(f"  {asset:<14} → {count} records upserted")
    log.info("=" * 55)


# ─────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventOracle Market Data Pipeline")
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        choices=list(ASSETS.keys()),
        help="Run pipeline for a single asset (default: all assets)",
    )
    args = parser.parse_args()

    if args.asset:
        selected = {args.asset: ASSETS[args.asset]}
    else:
        selected = ASSETS

    run_pipeline(selected)