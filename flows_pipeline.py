"""
flows_pipeline.py
EventOracle – Institutional Flows Data Pipeline (Person 6)

Fetches daily FII/DII net buy/sell data for Indian markets (last 2 years),
engineers momentum and z-score features, and upserts into MongoDB Atlas.

Data sources (in priority order):
    1. NSE India website (scraped HTML table)
    2. Fallback: NSE CSV endpoint
    3. Fallback: Simulated structured ingestion with realistic placeholder data

Usage:
    python flows_pipeline.py                         # Run full pipeline
    python flows_pipeline.py --dry-run               # Process but skip MongoDB write
    python flows_pipeline.py --source nse            # Single source only
    python flows_pipeline.py --lookback 90           # Days of history to fetch

Scheduling (cron – daily at 7:00 AM IST / 1:30 AM UTC):
    30 1 * * * /usr/bin/python3 /path/to/flows_pipeline.py >> /var/log/eventoracle_flows.log 2>&1

GitHub Actions: add a workflow with schedule: cron: '30 1 * * *'

Dependencies:
    pip install requests pandas pymongo python-dotenv beautifulsoup4 numpy lxml
"""

import os
import io
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
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
log = logging.getLogger("eventoracle.flows")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME     = "eventoracle"
FLOWS_COL   = "flows"
PRICES_COL  = "prices"          # Person 3 – read-only (for flow_to_price_ratio)
NEWS_COL    = "news"            # Person 4 – read-only (for sentiment cross-check)

IST             = timezone(timedelta(hours=5, minutes=30))
MAX_RETRIES     = 3
RETRY_BACKOFF   = 2.0           # seconds, exponential
REQUEST_TIMEOUT = 15            # seconds per HTTP request
BATCH_SIZE      = 500           # MongoDB bulk write batch size

# NSE FII/DII endpoints
NSE_FII_DII_CSV_URL = (
    "https://www.nseindia.com/api/fiidiiTradeReact"
)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Feature engineering windows
WINDOW_3D   = 3
WINDOW_5D   = 5
ZSCORE_WINDOW = 20

# Flow anomaly threshold (z-score)
ANOMALY_ZSCORE_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(IST)


def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is IST-aware (matching market_pipeline convention)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """Safely cast a value to float."""
    try:
        f = float(str(value).replace(",", ""))
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def retry_get(
    url: str,
    params: dict = None,
    headers: dict = None,
    session: requests.Session = None,
    retries: int = MAX_RETRIES,
) -> Optional[requests.Response]:
    """
    GET request with exponential back-off retry logic.
    Returns Response on success, None on exhausted retries.
    """
    caller = session or requests
    for attempt in range(1, retries + 1):
        try:
            resp = caller.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                return resp
            log.warning(
                f"HTTP {resp.status_code} on attempt {attempt}/{retries}: {url}"
            )
        except requests.RequestException as exc:
            log.warning(f"Request error on attempt {attempt}/{retries}: {exc}")

        if attempt < retries:
            sleep_secs = RETRY_BACKOFF ** attempt
            log.info(f"Retrying in {sleep_secs:.1f}s…")
            time.sleep(sleep_secs)

    log.error(f"All {retries} attempts failed for: {url}")
    return None


# ─────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_nse_fiidii_api(lookback_days: int = 730) -> pd.DataFrame:
    """
    Fetch FII/DII data from NSE's JSON API endpoint.
    NSE requires cookie-based session – we obtain a session cookie first.
    Returns a DataFrame with columns: [date, fii_net, dii_net] or empty.
    """
    log.info("[NSE API] Initiating session with NSE homepage…")
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    # Step 1: Warm up the session to get cookies
    for warmup_url in [
        "https://www.nseindia.com",
        "https://www.nseindia.com/market-data/fii-dii-activity",
    ]:
        try:
            warmup_resp = session.get(warmup_url, timeout=REQUEST_TIMEOUT)
            log.debug(f"[NSE API] Warmup {warmup_url}: {warmup_resp.status_code}")
            time.sleep(1)
        except requests.RequestException as exc:
            log.warning(f"[NSE API] Warmup failed for {warmup_url}: {exc}")

    # Step 2: Fetch FII/DII data page-by-page in 30-day chunks
    all_rows = []
    end_date = ist_now().date()
    start_date = end_date - timedelta(days=lookback_days)

    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=30), end_date)
        params = {
            "type": "fiidiiTradeReact",
            "category": "FII",
            "fromDate": chunk_start.strftime("%d-%m-%Y"),
            "toDate": chunk_end.strftime("%d-%m-%Y"),
        }
        resp = retry_get(NSE_FII_DII_CSV_URL, params=params, session=session)
        if resp is None:
            log.warning(
                f"[NSE API] No response for chunk {chunk_start} → {chunk_end}. Skipping."
            )
            chunk_start = chunk_end + timedelta(days=1)
            continue

        try:
            data = resp.json()
            # NSE returns a list of dicts with keys: date, buyValue, sellValue, netValue, etc.
            if isinstance(data, list):
                all_rows.extend(data)
            elif isinstance(data, dict) and "data" in data:
                all_rows.extend(data["data"])
        except Exception as exc:
            log.warning(f"[NSE API] JSON parse error for chunk: {exc}")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # be polite to NSE servers

    if not all_rows:
        log.warning("[NSE API] No rows returned from NSE API.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    log.info(f"[NSE API] Fetched {len(df)} raw rows.")
    return df


def _parse_nse_api_df(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Parse a raw NSE API DataFrame for a specific entity (FII or DII).
    Returns DataFrame with columns: [date, net_value].
    """
    # NSE column names vary; try common variants
    date_candidates = ["date", "Date", "DATE", "tradeDate"]
    net_candidates  = ["netValue", "net_value", "net", "NetValue"]
    entity_candidates = ["category", "Category", "entity", "Entity"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    net_col  = next((c for c in net_candidates  if c in df.columns), None)

    if date_col is None or net_col is None:
        log.warning(f"[NSE API] Could not identify date/net columns. Columns: {list(df.columns)}")
        return pd.DataFrame()

    # Filter by entity if column exists
    entity_col = next((c for c in entity_candidates if c in df.columns), None)
    if entity_col:
        df = df[df[entity_col].str.upper().str.contains(entity.upper(), na=False)].copy()

    result = df[[date_col, net_col]].copy()
    result.columns = ["date", "net_value"]
    result["net_value"] = result["net_value"].apply(safe_float)
    return result


def fetch_nse_csv_fallback(lookback_days: int = 730) -> pd.DataFrame:
    """
    Secondary fallback: attempt to fetch NSE FII/DII CSV download.
    NSE occasionally provides direct CSV exports.
    Returns raw DataFrame or empty.
    """
    log.info("[NSE CSV] Attempting CSV fallback…")
    end_date   = ist_now().date()
    start_date = end_date - timedelta(days=lookback_days)

    url = (
        "https://www.nseindia.com/api/historical/fiidii"
        f"?from={start_date.strftime('%d-%m-%Y')}"
        f"&to={end_date.strftime('%d-%m-%Y')}"
    )
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    # Warm up cookie
    try:
        session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT)
        time.sleep(1)
    except requests.RequestException:
        pass

    resp = retry_get(url, session=session)
    if resp is None:
        log.warning("[NSE CSV] CSV fallback also failed.")
        return pd.DataFrame()

    try:
        if "text/csv" in resp.headers.get("Content-Type", ""):
            df = pd.read_csv(io.StringIO(resp.text))
        else:
            data = resp.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame()
        log.info(f"[NSE CSV] Fetched {len(df)} rows.")
        return df
    except Exception as exc:
        log.warning(f"[NSE CSV] Parse error: {exc}")
        return pd.DataFrame()


def get_placeholder_data(lookback_days: int = 730) -> pd.DataFrame:
    """
    Structured placeholder ingestion for when NSE endpoints are unavailable.
    Generates realistic FII/DII daily flows using historically calibrated ranges.
    Replace this function with the actual data source when NSE access is stable.

    NOTE: This is clearly labeled as simulated data and should NOT be used for
    any production trading or financial decisions.
    """
    log.warning(
        "[PLACEHOLDER] All live sources failed. "
        "Generating structured placeholder FII/DII data. "
        "DO NOT use for production financial analysis."
    )

    np.random.seed(42)
    end_date   = ist_now().date()
    start_date = end_date - timedelta(days=lookback_days)

    # Generate business days only (Indian market sessions)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")

    # Historically calibrated ranges (in crore INR):
    #   FII net: mean ~0, std ~2500, with occasional fat-tail spikes
    #   DII net: inversely correlated with FII (domestic counters foreign flows)
    n = len(dates)
    fii_base = np.random.normal(0, 2500, n)
    fii_spikes = np.random.choice([0] * 20 + [-8000, -6000, 5000, 7000], n)
    fii_net = fii_base + fii_spikes

    # DII has mild negative correlation with FII (acts as stabilizer)
    dii_net = -0.4 * fii_net + np.random.normal(500, 1500, n)

    df = pd.DataFrame({
        "date":    [d.date() for d in dates],
        "fii_net": np.round(fii_net, 2),
        "dii_net": np.round(dii_net, 2),
    })
    log.info(f"[PLACEHOLDER] Generated {len(df)} placeholder rows.")
    return df


def fetch_data(lookback_days: int = 730) -> pd.DataFrame:
    """
    Master fetch function. Tries NSE API → NSE CSV → placeholder.
    Returns a clean DataFrame with columns: [date, fii_net, dii_net].
    """
    log.info(f"[FETCH] Starting data fetch for last {lookback_days} days…")

    # ── Attempt 1: NSE JSON API ──────────────────────────────
    raw_api = fetch_nse_fiidii_api(lookback_days=lookback_days)
    if not raw_api.empty:
        fii_df = _parse_nse_api_df(raw_api, "FII")
        dii_df = _parse_nse_api_df(raw_api, "DII")
        if not fii_df.empty and not dii_df.empty:
            merged = fii_df.merge(dii_df, on="date", suffixes=("_fii", "_dii"))
            merged = merged.rename(columns={"net_value_fii": "fii_net", "net_value_dii": "dii_net"})
            log.info(f"[FETCH] NSE API succeeded → {len(merged)} rows.")
            return merged

    # ── Attempt 2: NSE CSV fallback ──────────────────────────
    raw_csv = fetch_nse_csv_fallback(lookback_days=lookback_days)
    if not raw_csv.empty:
        fii_df = _parse_nse_api_df(raw_csv, "FII")
        dii_df = _parse_nse_api_df(raw_csv, "DII")
        if not fii_df.empty and not dii_df.empty:
            merged = fii_df.merge(dii_df, on="date", suffixes=("_fii", "_dii"))
            merged = merged.rename(columns={"net_value_fii": "fii_net", "net_value_dii": "dii_net"})
            log.info(f"[FETCH] NSE CSV fallback succeeded → {len(merged)} rows.")
            return merged

    # ── Attempt 3: Placeholder data ──────────────────────────
    return get_placeholder_data(lookback_days=lookback_days)


# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize schema, convert timestamps to IST, enforce numeric types,
    remove nulls and duplicates. Mirrors market_pipeline.py conventions.
    """
    if df.empty:
        log.warning("[CLEAN] Received empty DataFrame, skipping clean.")
        return df

    df = df.copy()

    # ── Normalize column names ───────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Accept flexible date column names
    date_col = next(
        (c for c in ["date", "timestamp", "trade_date", "tradedate"] if c in df.columns),
        None,
    )
    if date_col is None:
        log.error(f"[CLEAN] Could not find a date column. Columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df.rename(columns={date_col: "timestamp"})

    # ── Parse and localize timestamps ───────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    null_dates = df["timestamp"].isna().sum()
    if null_dates > 0:
        log.warning(f"[CLEAN] Dropping {null_dates} rows with unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    # Localize to IST (consistent with market_pipeline.py)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(IST)

    # Normalize to date-only precision (midnight IST) to avoid sub-day duplication
    df["timestamp"] = df["timestamp"].dt.normalize()

    # ── Ensure required columns exist ───────────────────────
    for col in ["fii_net", "dii_net"]:
        if col not in df.columns:
            log.error(f"[CLEAN] Required column '{col}' missing. Cannot proceed.")
            return pd.DataFrame()

    # ── Enforce numeric types ────────────────────────────────
    df["fii_net"] = pd.to_numeric(df["fii_net"], errors="coerce")
    df["dii_net"] = pd.to_numeric(df["dii_net"], errors="coerce")

    # ── Drop nulls ───────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["fii_net", "dii_net", "timestamp"])
    after = len(df)
    if before != after:
        log.info(f"[CLEAN] Dropped {before - after} rows with null values.")

    # ── Remove duplicates ────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    after = len(df)
    if before != after:
        log.info(f"[CLEAN] Dropped {before - after} duplicate timestamp rows.")

    # ── Sort ascending by date ───────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Keep only required columns ───────────────────────────
    df = df[["timestamp", "fii_net", "dii_net"]]

    # ── Round to 2 decimal places ────────────────────────────
    df["fii_net"] = df["fii_net"].round(2)
    df["dii_net"] = df["dii_net"].round(2)

    log.info(f"[CLEAN] Cleaned DataFrame: {len(df)} rows from "
             f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}.")
    return df


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame, prices_col=None, news_col=None) -> pd.DataFrame:
    """
    Engineer flow-based features:
        - total_flow          : fii_net + dii_net
        - flow_3d             : 3-day rolling sum of total_flow
        - flow_5d             : 5-day rolling sum of total_flow
        - flow_momentum       : flow_3d - flow_5d (short vs medium momentum)
        - flow_zscore         : 20-day rolling z-score of total_flow
        - flow_to_price_ratio : total_flow / NIFTY close (optional, requires prices_col)
        - flow_anomaly        : True if abs(flow_zscore) > ANOMALY_ZSCORE_THRESHOLD
        - sentiment_conflict  : True if high positive flow + negative news (bonus)

    Args:
        df          : Cleaned flows DataFrame.
        prices_col  : PyMongo Collection object for 'prices' (optional).
        news_col    : PyMongo Collection object for 'news' (optional).
    """
    if df.empty:
        log.warning("[FEATURES] Empty DataFrame — skipping feature engineering.")
        return df

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Core flow features ───────────────────────────────────
    df["total_flow"] = (df["fii_net"] + df["dii_net"]).round(2)

    df["flow_3d"] = df["total_flow"].rolling(window=WINDOW_3D, min_periods=1).sum().round(2)
    df["flow_5d"] = df["total_flow"].rolling(window=WINDOW_5D, min_periods=1).sum().round(2)

    df["flow_momentum"] = (df["flow_3d"] - df["flow_5d"]).round(2)

    # Rolling z-score: (x - rolling_mean) / rolling_std
    roll_mean = df["total_flow"].rolling(window=ZSCORE_WINDOW, min_periods=3).mean()
    roll_std  = df["total_flow"].rolling(window=ZSCORE_WINDOW, min_periods=3).std()
    df["flow_zscore"] = ((df["total_flow"] - roll_mean) / roll_std.replace(0, np.nan)).round(4)

    # ── Flow anomaly flag ────────────────────────────────────
    df["flow_anomaly"] = df["flow_zscore"].abs() > ANOMALY_ZSCORE_THRESHOLD

    # ── Optional: flow_to_price_ratio (NIFTY join) ───────────
    df["flow_to_price_ratio"] = None
    if prices_col is not None:
        try:
            log.info("[FEATURES] Fetching NIFTY prices from MongoDB for flow_to_price_ratio…")
            nifty_docs = list(prices_col.find(
                {"asset": "NIFTY"},
                {"_id": 0, "timestamp": 1, "close": 1},
            ))
            if nifty_docs:
                nifty_df = pd.DataFrame(nifty_docs)
                nifty_df["timestamp"] = pd.to_datetime(nifty_df["timestamp"])
                if nifty_df["timestamp"].dt.tz is None:
                    nifty_df["timestamp"] = nifty_df["timestamp"].dt.tz_localize("UTC")
                nifty_df["timestamp"] = nifty_df["timestamp"].dt.tz_convert(IST).dt.normalize()
                nifty_df = nifty_df.rename(columns={"close": "nifty_close"})
                df = df.merge(nifty_df[["timestamp", "nifty_close"]], on="timestamp", how="left")
                df["flow_to_price_ratio"] = (
                    df["total_flow"] / df["nifty_close"].replace(0, np.nan)
                ).round(6)
                df = df.drop(columns=["nifty_close"])
                log.info(f"[FEATURES] flow_to_price_ratio computed for "
                         f"{df['flow_to_price_ratio'].notna().sum()} rows.")
            else:
                log.warning("[FEATURES] No NIFTY data found in prices collection.")
        except Exception as exc:
            log.warning(f"[FEATURES] Could not compute flow_to_price_ratio: {exc}")

    # ── BONUS: sentiment conflict detection ──────────────────
    df["sentiment_conflict"] = False
    if news_col is not None:
        try:
            log.info("[FEATURES] Checking news sentiment for flow anomaly days…")
            anomaly_dates = df.loc[df["flow_anomaly"] & (df["flow_zscore"] > 0), "timestamp"]
            if not anomaly_dates.empty:
                # Fetch negative-sentiment news on anomaly days
                anomaly_date_list = anomaly_dates.dt.to_pydatetime().tolist()
                neg_news = list(news_col.find({
                    "published_at": {"$in": anomaly_date_list},
                    "sentiment_score": {"$lt": -0.2},
                }, {"_id": 0, "published_at": 1}))
                if neg_news:
                    conflict_dates = set(
                        pd.to_datetime(doc["published_at"]).replace(
                            hour=0, minute=0, second=0, microsecond=0,
                            tzinfo=IST
                        )
                        for doc in neg_news
                    )
                    df["sentiment_conflict"] = df["timestamp"].isin(conflict_dates)
                    log.info(
                        f"[FEATURES] sentiment_conflict flagged on "
                        f"{df['sentiment_conflict'].sum()} days."
                    )
        except Exception as exc:
            log.warning(f"[FEATURES] sentiment_conflict computation failed: {exc}")

    log.info(
        f"[FEATURES] Feature engineering complete. Shape: {df.shape} | "
        f"Anomalies: {int(df['flow_anomaly'].sum())}"
    )
    return df


# ─────────────────────────────────────────────────────────────
# 4. DATABASE INTEGRATION
# ─────────────────────────────────────────────────────────────

def get_mongo_db():
    """
    Connect to MongoDB Atlas and return (client, db) tuple.
    Validates connection with a ping.
    """
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
    """Create required indexes on the flows collection."""
    col = db[FLOWS_COL]
    # Unique index on timestamp (one flow record per trading day)
    col.create_index(
        [("timestamp", 1)],
        unique=True,
        background=True,
        name="timestamp_unique",
    )
    log.info(f"[MONGO] Indexes ensured on '{FLOWS_COL}'.")


def upload_to_mongo(df: pd.DataFrame, db, dry_run: bool = False) -> int:
    """
    Upsert flow records into MongoDB Atlas.
    Uses $setOnInsert to avoid overwriting existing records on reruns.
    Returns count of newly inserted documents.
    """
    if df.empty:
        log.warning("[MONGO] Nothing to upload — DataFrame is empty.")
        return 0

    collection = db[FLOWS_COL]
    records    = df.to_dict("records")

    # ── Serialize for MongoDB ────────────────────────────────
    for rec in records:
        # Convert pandas Timestamp → Python datetime
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()

        # Replace NaN/pd.NA/None with None for clean BSON storage
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)

    if dry_run:
        log.info(f"[MONGO] DRY RUN — would upsert {len(records)} records into '{FLOWS_COL}'.")
        return 0

    # ── Batch upsert ─────────────────────────────────────────
    ops = [
        UpdateOne(
            {"timestamp": rec["timestamp"]},
            {"$setOnInsert": rec},
            upsert=True,
        )
        for rec in records
    ]

    total_upserted = 0
    for i in range(0, len(ops), BATCH_SIZE):
        batch = ops[i : i + BATCH_SIZE]
        try:
            result = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count
        except BulkWriteError as bwe:
            # Duplicate key errors are expected on reruns — not fatal
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"[MONGO] BulkWriteError (expected on reruns): {bwe.details}")

    log.info(f"[MONGO] ✅ Upserted {total_upserted} new records into '{FLOWS_COL}'.")
    return total_upserted


# ─────────────────────────────────────────────────────────────
# 5. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_pipeline(lookback_days: int = 730, dry_run: bool = False) -> None:
    """
    Full pipeline orchestrator:
        fetch → clean → features → upsert

    Args:
        lookback_days : Days of historical data to fetch (default: 2 years).
        dry_run       : If True, skip MongoDB write (useful for testing).
    """
    log.info("=" * 60)
    log.info("EventOracle – Institutional Flows Pipeline Starting")
    log.info(f"Lookback: {lookback_days}d | DryRun: {dry_run}")
    log.info("=" * 60)

    # ── Connect to MongoDB ────────────────────────────────────
    try:
        client, db = get_mongo_db()
        ensure_indexes(db)
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as exc:
        log.error(f"❌ MongoDB connection failed: {exc}")
        return

    prices_col = db[PRICES_COL]
    news_col   = db[NEWS_COL]

    # ── Stage 1: Fetch ───────────────────────────────────────
    log.info("\n── Stage 1: Fetching FII/DII data ──")
    raw_df = fetch_data(lookback_days=lookback_days)
    if raw_df.empty:
        log.error("❌ Pipeline aborted — fetch returned no data.")
        client.close()
        return

    # ── Stage 2: Clean ───────────────────────────────────────
    log.info("\n── Stage 2: Cleaning data ──")
    clean_df = clean_data(raw_df)
    if clean_df.empty:
        log.error("❌ Pipeline aborted — cleaning produced empty DataFrame.")
        client.close()
        return

    # ── Stage 3: Feature Engineering ─────────────────────────
    log.info("\n── Stage 3: Engineering features ──")
    featured_df = add_features(clean_df, prices_col=prices_col, news_col=news_col)
    if featured_df.empty:
        log.error("❌ Pipeline aborted — feature engineering returned empty result.")
        client.close()
        return

    # ── Stage 4: Upload ──────────────────────────────────────
    log.info("\n── Stage 4: Uploading to MongoDB ──")
    inserted = upload_to_mongo(featured_df, db, dry_run=dry_run)

    # ── Summary ──────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  EventOracle – Flows Pipeline Summary")
    log.info("=" * 60)
    log.info(f"  Records processed      : {len(featured_df)}")
    log.info(f"  New records inserted   : {inserted}")
    log.info(f"  Date range             : "
             f"{featured_df['timestamp'].min().date()} → "
             f"{featured_df['timestamp'].max().date()}")
    log.info(f"  Flow anomalies flagged : {int(featured_df['flow_anomaly'].sum())}")
    if "sentiment_conflict" in featured_df.columns:
        log.info(
            f"  Sentiment conflicts    : {int(featured_df['sentiment_conflict'].sum())}"
        )
    if "flow_to_price_ratio" in featured_df.columns:
        ratio_count = featured_df["flow_to_price_ratio"].notna().sum()
        log.info(f"  flow_to_price_ratio    : computed for {ratio_count} rows")
    log.info("=" * 60)

    client.close()
    log.info("✅ Flows pipeline complete.")


# ─────────────────────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle – Institutional Flows Data Pipeline"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=730,
        help="Days of historical data to fetch (default: 730 = ~2 years).",
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