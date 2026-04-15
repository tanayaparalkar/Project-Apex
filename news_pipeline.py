"""
EventOracle: News and Event Intelligence Pipeline for Indian Markets
Author: Person 4
Description:
    - Fetch financial news from Indian market sources
    - Perform sentiment analysis using FinBERT (fallback: VADER)
    - Tag events using keyword-based logic
    - Store structured data in MongoDB Atlas
"""

import feedparser
import pandas as pd
import re
import logging
import pytz
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import argparse

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
MONGO_URI = "YOUR_MONGODB_ATLAS_CONNECTION_STRING"
DB_NAME = "eventoracle"
COLLECTION_NAME = "news"

RSS_FEEDS = {
    "Economic Times Markets": "[economictimes.indiatimes.com](https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms)",
    "Mint": "[livemint.com](https://www.livemint.com/rss/markets)",
    "Business Standard": "[business-standard.com](https://www.business-standard.com/rss/markets-106.rss)",
}

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------
# LOGGING
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)

# ---------------------------------------------
# FETCH NEWS
# ---------------------------------------------
def fetch_news(limit=None):
    all_news = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:limit] if limit else feed.entries
            for e in entries:
                all_news.append({
                    "headline": e.title,
                    "source": source,
                    "published": e.get("published", datetime.utcnow().isoformat()),
                    "link": e.link,
                })
        except Exception as err:
            logging.error(f"Error fetching from {source}: {err}")
    df = pd.DataFrame(all_news)
    logging.info(f"Fetched {len(df)} total articles")
    return df


# ---------------------------------------------
# DATA CLEANING
# ---------------------------------------------
def clean_news(df):
    if df.empty:
        return df
    df.drop_duplicates(subset=["headline", "published"], inplace=True)
    
    # Normalize timestamp
    def normalize_ts(ts):
        try:
            dt = datetime.strptime(ts[:25], "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                dt = datetime.utcnow()
        return dt.astimezone(IST).isoformat()
    
    df["published"] = df["published"].apply(normalize_ts)
    df["headline_clean"] = df["headline"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
    return df


# ---------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------
class SentimentAnalyzer:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.use_finbert = True
            logging.info("Using FinBERT for sentiment analysis")
        except Exception:
            logging.warning("Falling back to VADER sentiment model")
            self.use_finbert = False
            self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text):
        if self.use_finbert:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1)
                label_id = torch.argmax(probs).item()
                sentiment_label = self.model.config.id2label[label_id]
                sentiment_score = probs[0][label_id].item()
        else:
            score = self.vader.polarity_scores(text)
            sentiment_label = (
                "positive" if score["compound"] > 0.05 else
                "negative" if score["compound"] < -0.05 else
                "neutral"
            )
            sentiment_score = score["compound"]
        return sentiment_label.lower(), float(sentiment_score)


def analyze_sentiment(df):
    if df.empty:
        return df
    analyzer = SentimentAnalyzer()
    sentiments = df["headline_clean"].apply(analyzer.analyze)
    df["sentiment_label"] = sentiments.apply(lambda x: x[0])
    df["sentiment_score"] = sentiments.apply(lambda x: x[1])
    return df


# ---------------------------------------------
# EVENT TAGGING
# ---------------------------------------------
KEYWORDS = {
    "Fed": ["fed", "powell", "fomc"],
    "CPI": ["inflation", "cpi", "consumer price"],
    "Oil": ["oil", "brent", "opec", "crude"],
    "Geopolitical": ["war", "conflict", "israel", "ukraine", "china", "border"],
    "RBI": ["rbi", "repo rate", "governor shaktikanta", "monetary policy"],
    "SEBI": ["sebi", "regulator", "ipo guidelines"],
    "FII": ["fii", "foreign investor", "foreign inflow", "outflow"],
    "GDP/IIP": ["gdp", "growth rate", "iip", "industrial output"],
}

def tag_events(df):
    def detect_event(text):
        for tag, keywords in KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return tag
        return "Other"
    df["event_tag"] = df["headline_clean"].apply(detect_event)
    return df


# ---------------------------------------------
# DATABASE STORAGE
# ---------------------------------------------
def upload_to_mongo(df):
    if df.empty:
        logging.info("No new data to upload.")
        return 0
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    operations = []
    for _, row in df.iterrows():
        operations.append(
            UpdateOne(
                {
                    "headline": row["headline"],
                    "published": row["published"],
                },
                {"$set": row.to_dict()},
                upsert=True
            )
        )
    if operations:
        result = col.bulk_write(operations)
        inserted = result.upserted_count + result.modified_count
        logging.info(f"Inserted/Updated {inserted} records into MongoDB.")
        return inserted
    return 0


# ---------------------------------------------
# PIPELINE RUNNER
# ---------------------------------------------
def run_pipeline(limit=None):
    df = fetch_news(limit)
    df = clean_news(df)
    df = analyze_sentiment(df)
    df = tag_events(df)
    upload_to_mongo(df)


# ---------------------------------------------
# CLI
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventOracle News Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of news per source", default=None)
    args = parser.parse_args()
    run_pipeline(limit=args.limit)
