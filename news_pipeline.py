"""
news_pipeline.py
EventOracle – News Ingestion & Intelligence Pipeline (Person 4)

Fetches financial news from RSS feeds, performs sentiment analysis,
tags events, and stores structured data into MongoDB Atlas.

Usage:
    python news_pipeline.py                          # Run all sources
    python news_pipeline.py --source et              # Single source
    python news_pipeline.py --limit 50               # Limit articles
    python news_pipeline.py --source mint --limit 20 # Combined

Scheduling (cron – daily at 7:00 AM IST / 1:30 AM UTC):
    30 1 * * * /usr/bin/python3 /path/to/news_pipeline.py >> /var/log/eventoracle_news.log 2>&1

Dependencies:
    pip install feedparser pymongo python-dotenv nltk vaderSentiment pandas pytz
"""

import os
import time
import re
import logging
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter

import requests
import feedparser
import pandas as pd
import pytz
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────

load_dotenv()

nltk.download("vader_lexicon", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.news")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI    = os.getenv("MONGO_URI")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
DB_NAME      = "eventoracle"
COLLECTION   = "news"
IST          = pytz.timezone("Asia/Kolkata")
BATCH_SIZE   = 200
MAX_RETRIES  = 3
RETRY_DELAY  = 3

RSS_SOURCES = {
    "et": {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    },
    "mint": {
        "name": "Mint",
        "url": "https://www.livemint.com/rss/markets",
    },
    "bs": {
        "name": "Business Standard",
        "url": "https://www.business-standard.com/rss/markets-106.rss",
    },
}

# Event tagging keywords (priority order matters — more specific first)
EVENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("RBI",          ["rbi", "reserve bank", "repo rate", "monetary policy", "mpc", "shaktikanta", "das"]),
    ("Fed",          ["federal reserve", "fed rate", "fomc", "jerome powell", "fed meeting", "us rate", "rate hike", "rate cut"]),
    ("CPI",          ["cpi", "inflation", "consumer price", "wpi", "wholesale price", "price index"]),
    ("Oil",          ["oil", "brent", "crude", "opec", "petroleum", "fuel price"]),
    ("Gold",         ["gold", "xauusd", "gold price", "gold futures", "gold etf", "yellow metal", "bullion"]),
    ("Silver",       ["silver", "xagusd", "silver price", "silver futures"]),
    ("Platinum",     ["platinum", "xptusd", "platinum price", "pgm"]),
    ("Bitcoin",      ["bitcoin", "btc", "crypto", "cryptocurrency", "blockchain", "ethereum", "digital currency"]),
    ("Nifty",        ["nifty", "sensex", "nse", "bse", "bank nifty", "banknifty", "nifty it", "nifty metal"]),
    ("Currency",     ["rupee", "inr", "dollar", "usd", "forex", "currency", "exchange rate", "inrusd"]),
    ("FII",          ["fii", "dii", "foreign institutional", "domestic institutional", "fpi", "foreign portfolio"]),
    ("GDP/IIP",      ["gdp", "iip", "gross domestic", "industrial production", "economic growth", "growth rate"]),
    ("SEBI",         ["sebi", "securities board", "market regulator", "circuit breaker", "f&o ban", "insider trading"]),
    ("Geopolitical", ["war", "sanction", "geopolit", "tension", "conflict", "tariff", "trade war", "election", "ceasefire"]),
]

# Google News RSS search queries — covers all tracked assets
GOOGLE_NEWS_QUERIES = [
    # Market indices (from market_pipeline)
    "Nifty 50", "Sensex", "Bank Nifty", "Nifty IT", "Nifty Metal",
    # Commodities
    "Brent crude oil", "gold price", "silver price", "platinum price",
    # Crypto
    "Bitcoin BTC",
    # Currency
    "Indian rupee USD",
    # Macro / policy
    "RBI monetary policy", "India inflation CPI", "India GDP growth",
    "Federal Reserve rate decision", "FII DII India",
    "SEBI regulation", "India stock market",
]

# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def fetch_news(sources: dict = None, limit: int = None) -> pd.DataFrame:
    """
    Fetch articles from RSS feeds.
    Returns a DataFrame with columns: headline, source, published_raw, link.
    """
    if sources is None:
        sources = RSS_SOURCES

    all_articles = []

    for key, meta in sources.items():
        log.info(f"[{key.upper()}] Fetching from: {meta['url']}")
        try:
            feed = feedparser.parse(meta["url"])

            if feed.bozo and feed.bozo_exception:
                log.warning(f"[{key.upper()}] Feed parse warning: {feed.bozo_exception}")

            entries = feed.entries
            if limit:
                entries = entries[:limit]

            for entry in entries:
                headline = (
                    entry.get("title", "")
                    or entry.get("summary", "")
                ).strip()

                if not headline:
                    continue

                # Try multiple timestamp fields
                published_raw = (
                    entry.get("published")
                    or entry.get("updated")
                    or entry.get("dc_date")
                    or ""
                )

                link = entry.get("link", "").strip()

                all_articles.append({
                    "headline":      headline,
                    "source":        meta["name"],
                    "published_raw": published_raw,
                    "link":          link,
                })

            log.info(f"[{key.upper()}] Fetched {len(entries)} articles.")

        except Exception as e:
            log.error(f"[{key.upper()}] Failed to fetch feed: {e}")

    if not all_articles:
        log.warning("No articles fetched from any source.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    log.info(f"Total raw articles fetched: {len(df)}")
    return df


def _http_get(url: str, params: dict = None, timeout: int = 15):
    """GET with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"  HTTP attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def fetch_google_news_historical(queries: list[str] = None,
                                  start_date: str = "2024-04-15",
                                  end_date: str = None) -> list[dict]:
    """
    Fetch historical news via Google News RSS (free, no API key needed).
    Iterates month-by-month for each query to get broad coverage.
    """
    if queries is None:
        queries = GOOGLE_NEWS_QUERIES
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_articles = []
    total_queries = len(queries)

    for qi, query in enumerate(queries, 1):
        # Walk month-by-month
        cursor = datetime(start_dt.year, start_dt.month, 1)
        while cursor <= end_dt:
            m_start = max(cursor, start_dt)
            if cursor.month == 12:
                m_end_dt = datetime(cursor.year + 1, 1, 1) - timedelta(days=1)
            else:
                m_end_dt = datetime(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
            m_end = min(m_end_dt, end_dt)

            after_str = m_start.strftime("%Y-%m-%d")
            before_str = m_end.strftime("%Y-%m-%d")

            url = (
                f"https://news.google.com/rss/search?"
                f"q={requests.utils.quote(query)}+after:{after_str}+before:{before_str}"
                f"&hl=en-IN&gl=IN&ceid=IN:en"
            )

            log.info(f"  [{qi}/{total_queries}] GoogleNews: '{query}' {after_str} → {before_str}")
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    headline = (entry.get("title", "") or "").strip()
                    if not headline:
                        continue
                    all_articles.append({
                        "headline":      headline,
                        "source":        f"GoogleNews",
                        "published_raw": entry.get("published", ""),
                        "link":          entry.get("link", ""),
                    })
            except Exception as e:
                log.warning(f"  GoogleNews fetch error: {e}")

            time.sleep(0.5)  # polite delay
            # Advance to next month
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1)

    log.info(f"[GoogleNews] Fetched {len(all_articles)} total historical articles.")
    return all_articles


def fetch_gnews_historical(queries: list[str] = None,
                            start_date: str = "2024-04-15",
                            end_date: str = None) -> list[dict]:
    """
    Fetch historical news from GNews API (free tier: 100 req/day).
    Supplements Google News with an additional source.
    """
    if not GNEWS_API_KEY:
        log.warning("[GNEWS] No API key set (GNEWS_API_KEY). Skipping.")
        return []
    if queries is None:
        queries = GOOGLE_NEWS_QUERIES
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_articles = []
    cursor = datetime(start_dt.year, start_dt.month, 1)
    rate_limited = False

    while cursor <= end_dt and not rate_limited:
        m_start = max(cursor, start_dt)
        if cursor.month == 12:
            m_end_dt = datetime(cursor.year + 1, 1, 1) - timedelta(days=1)
        else:
            m_end_dt = datetime(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
        m_end = min(m_end_dt, end_dt)

        from_str = m_start.strftime("%Y-%m-%dT00:00:00Z")
        to_str = m_end.strftime("%Y-%m-%dT23:59:59Z")

        for query in queries:
            params = {
                "q": query, "lang": "en", "country": "in",
                "from": from_str, "to": to_str,
                "max": 10, "apikey": GNEWS_API_KEY,
            }
            resp = _http_get("https://gnews.io/api/v4/search", params=params)
            if resp is None:
                # Check if we're rate-limited (all retries failed)
                rate_limited = True
                log.warning("[GNEWS] Rate limit hit (403). Stopping GNews fetch early.")
                break
            try:
                for art in resp.json().get("articles", []):
                    all_articles.append({
                        "headline":      art.get("title", "").strip(),
                        "source":        f"GNews-{art.get('source', {}).get('name', 'Unknown')}",
                        "published_raw": art.get("publishedAt", ""),
                        "link":          art.get("url", ""),
                    })
            except Exception as e:
                log.error(f"[GNEWS] JSON parse error: {e}")
            time.sleep(1)

        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)

    log.info(f"[GNEWS] Fetched {len(all_articles)} historical articles.")
    return all_articles


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def _parse_timestamp(raw: str) -> datetime | None:
    """Parse a raw timestamp string into an IST-aware datetime."""
    if not raw:
        return None

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(IST)
        except ValueError:
            continue

    # Fallback: use current time in IST
    log.debug(f"Could not parse timestamp: '{raw}' — using current time.")
    return datetime.now(IST)


def _clean_text(text: str) -> str:
    """Lowercase and strip special characters from text."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\.\,\%\$\&\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Parse and normalize timestamps to IST
    - Clean headline text
    - Deduplicate on (headline + timestamp)
    """
    if df.empty:
        return df

    # Parse timestamps
    df["timestamp"] = df["published_raw"].apply(_parse_timestamp)

    # Clean headline
    df["headline_clean"] = df["headline"].apply(_clean_text)

    # Drop rows without a timestamp or headline
    before = len(df)
    df = df.dropna(subset=["timestamp", "headline"])
    df = df[df["headline"].str.strip() != ""]

    # Deduplicate on (headline_clean + date portion of timestamp)
    df["_dedup_key"] = df["headline_clean"] + "|" + df["timestamp"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M") if t else ""
    )
    df = df.drop_duplicates(subset=["_dedup_key"])
    df = df.drop(columns=["_dedup_key", "published_raw"])

    after = len(df)
    if before != after:
        log.info(f"Dropped {before - after} duplicate/null articles. Remaining: {after}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()


def _vader_sentiment(text: str) -> tuple[float, str]:
    """Return (compound_score, label) using VADER."""
    scores = _vader.polarity_scores(text)
    compound = round(scores["compound"], 4)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment_score and sentiment_label columns using VADER."""
    if df.empty:
        return df

    results = df["headline_clean"].apply(_vader_sentiment)
    df["sentiment_score"] = results.apply(lambda x: x[0])
    df["sentiment_label"] = results.apply(lambda x: x[1])

    dist = df["sentiment_label"].value_counts().to_dict()
    log.info(f"Sentiment distribution: {dist}")
    return df


# ─────────────────────────────────────────────
# 4. EVENT TAGGING
# ─────────────────────────────────────────────

def _tag_event(text: str) -> str:
    """Return the first matching event category or 'Other'."""
    for category, keywords in EVENT_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return category
    return "Other"


def tag_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_type column by keyword matching on cleaned headline."""
    if df.empty:
        return df

    df["event_type"] = df["headline_clean"].apply(_tag_event)

    dist = df["event_type"].value_counts().to_dict()
    log.info(f"Event tag distribution: {dist}")
    return df


# ─────────────────────────────────────────────
# 5. NEWS VELOCITY & ANOMALY FLAG
# ─────────────────────────────────────────────

def add_velocity_and_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    - news_velocity: article count per source per calendar day
    - anomaly_flag: 1 if that source's daily count > mean + 2*std, else 0
    """
    if df.empty:
        return df

    df["date"] = df["timestamp"].apply(lambda t: t.date())

    velocity = (
        df.groupby(["source", "date"])
        .size()
        .reset_index(name="news_velocity")
    )

    # Compute per-source stats for anomaly detection
    stats = (
        velocity.groupby("source")["news_velocity"]
        .agg(["mean", "std"])
        .reset_index()
    )
    velocity = velocity.merge(stats, on="source")
    velocity["threshold"] = velocity["mean"] + 2 * velocity["std"].fillna(0)
    velocity["anomaly_flag"] = (velocity["news_velocity"] > velocity["threshold"]).astype(int)
    velocity = velocity[["source", "date", "news_velocity", "anomaly_flag"]]

    df = df.merge(velocity, on=["source", "date"], how="left")
    df["news_velocity"] = df["news_velocity"].fillna(1).astype(int)
    df["anomaly_flag"]  = df["anomaly_flag"].fillna(0).astype(int)
    df = df.drop(columns=["date"])

    spike_count = df["anomaly_flag"].sum()
    if spike_count:
        log.warning(f"⚠️  Anomaly flag set on {spike_count} articles (news velocity spike).")

    return df


# ─────────────────────────────────────────────
# 6. DATABASE STORAGE
# ─────────────────────────────────────────────

def get_mongo_collection():
    """Connect to MongoDB Atlas and return the news collection."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set in environment variables.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]
    col.create_index(
        [("headline", 1), ("timestamp", 1)],
        unique=True,
        background=True,
        name="headline_timestamp_unique",
    )
    return col


def upload_to_mongo(df: pd.DataFrame, collection) -> int:
    """
    Upsert cleaned, enriched articles into MongoDB.
    Returns count of newly inserted documents.
    """
    if df.empty:
        log.warning("Nothing to upload — dataframe is empty.")
        return 0

    # Final column selection matching schema.json
    output_cols = [
        "timestamp", "headline", "source", "link",
        "sentiment_score", "sentiment_label",
        "event_type", "news_velocity", "anomaly_flag",
    ]
    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    records = df[output_cols].to_dict("records")

    # Convert pandas Timestamps → Python datetime
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()

    ops = [
        UpdateOne(
            {"headline": rec["headline"], "timestamp": rec["timestamp"]},
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
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"BulkWriteError (expected on reruns): {bwe.details}")

    log.info(f"✅ Upserted {total_upserted} new articles into '{COLLECTION}'.")
    return total_upserted


# ─────────────────────────────────────────────
# 7. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(sources: dict = None, limit: int = None, historical: bool = False):
    """Run the full news ingestion and enrichment pipeline."""
    mode = "historical (2024-2025)" if historical else "live RSS"
    log.info("=" * 55)
    log.info("EventOracle – News Intelligence Pipeline Starting")
    log.info(f"Mode: {mode} | Sources: {list((sources or RSS_SOURCES).keys())} | Limit: {limit or 'all'}")
    log.info("=" * 55)

    try:
        collection = get_mongo_collection()
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as e:
        log.error(f"❌ MongoDB connection failed: {e}")
        return

    # Step 1: Fetch
    all_raw: list[dict] = []

    # Always try RSS feeds for recent news
    df_rss = fetch_news(sources=sources, limit=limit)
    if not df_rss.empty:
        all_raw.extend(df_rss.to_dict("records"))

    # Historical mode: Google News RSS + GNews API (no hardcoded data)
    if historical:
        log.info("\n── Historical: Google News RSS (primary) ──")
        google_articles = fetch_google_news_historical()
        all_raw.extend(google_articles)

        log.info("\n── Historical: GNews API (supplementary) ──")
        gnews_articles = fetch_gnews_historical()
        all_raw.extend(gnews_articles)

    if not all_raw:
        log.error("Pipeline aborted — no articles fetched.")
        return

    df = pd.DataFrame(all_raw)
    log.info(f"Total raw articles collected: {len(df)}")

    # Step 2: Clean
    df = clean_news(df)
    if df.empty:
        log.error("Pipeline aborted — all articles dropped during cleaning.")
        return

    # Step 3: Sentiment
    df = analyze_sentiment(df)

    # Step 4: Event tagging
    df = tag_events(df)

    # Step 5: Velocity + anomaly
    df = add_velocity_and_anomaly(df)

    # Step 6: Upload
    inserted = upload_to_mongo(df, collection)

    log.info("=" * 55)
    log.info("Pipeline Complete")
    log.info(f"  Articles processed : {len(df)}")
    log.info(f"  New records inserted: {inserted}")
    log.info("=" * 55)


# ─────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventOracle News Intelligence Pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=list(RSS_SOURCES.keys()),
        help="Fetch from a single source key: et | mint | bs (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to fetch per source (default: all available)",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        default=False,
        help="Fetch historical news from Apr 2024 via Google News RSS + GNews API",
    )
    args = parser.parse_args()

    selected_sources = {args.source: RSS_SOURCES[args.source]} if args.source else RSS_SOURCES
    run_pipeline(sources=selected_sources, limit=args.limit, historical=args.historical)
