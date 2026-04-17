"""
market_agent.py
EventOracle – LLM-Based Market Agent

Uses Google's Gemini to synthesize data from the prediction, social anomaly,
and raw feature pipelines to provide an actionable trading recommendation.

Usage:
    python market_agent.py --asset NIFTY
"""

import argparse
import logging
import os
import sys
from datetime import timedelta, timezone

from google import genai
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.market_agent")

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "eventoracle"
IST = timezone(timedelta(hours=5, minutes=30))

KNOWN_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]

def get_mongo_client() -> MongoClient:
    if not MONGO_URI:
        log.error("MONGO_URI not set in environment.")
        sys.exit(1)
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client

def fetch_latest_doc(client: MongoClient, collection: str, asset: str) -> dict:
    col = client[DB_NAME][collection]
    # Sort by timestamp descending to get the most recent record
    doc = col.find_one({"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0})
    return doc or {}

def format_system_prompt(asset: str, feat_doc: dict, soc_doc: dict, pred_doc: dict) -> str:
    """Construct a detailed prompt using the fetched data."""
    
    # Extract values safely
    timestamp = pred_doc.get("timestamp") or feat_doc.get("timestamp")
    
    direction = pred_doc.get("predicted_direction", "N/A")
    magnitude = pred_doc.get("predicted_magnitude", "N/A")
    confidence = pred_doc.get("confidence_score", "N/A")
    regime = pred_doc.get("regime", "N/A")
    
    social_score = soc_doc.get("social_sentiment_score", "N/A")
    social_buzz = soc_doc.get("social_buzz_intensity", "N/A")
    anomaly_flag = soc_doc.get("composite_anomaly_flag", "N/A")
    
    rsi = feat_doc.get("rsi", "N/A")
    volatility = feat_doc.get("rolling_volatility", "N/A")
    returns = feat_doc.get("returns", "N/A")

    prompt = f"""
You are the EventOracle Market Agent, an elite AI quantitative trading assistant.
Analyze the latest data pipeline outputs for the asset '{asset}' and provide a structured, highly actionable trading recommendation.

### Current State for {asset} (as of {timestamp})

**1. ML Predictive Model Output (LightGBM)**
- Predicted Direction: {direction} (-1: Down, 0: Neutral, 1: Up)
- Predicted Magnitude (Return %): {magnitude}
- Model Confidence Score: {confidence}
- Market Regime Detected: {regime} (0: Normal, 1: Volatile/Trending, 2: Mean-reverting)

**2. Social & Anomaly Proxy Output (Isolation Forest)**
- Social Sentiment Score: {social_score} (Positive > 0, Negative < 0)
- Social Buzz Intensity (Z-Score): {social_buzz}
- Composite Anomaly Flag: {anomaly_flag} (1: High Market Stress / Anomaly Detected, 0: Normal)

**3. Raw Technical & Macro Features**
- RSI (Relative Strength Index): {rsi}
- Rolling Volatility: {volatility}
- Recent Return: {returns}

### Your Task
Provide a cohesive, readable Markdown report with the following sections:
1. **Executive Summary**: A one-sentence bottom line.
2. **Signal Synthesis**: Interpret how the ML prediction aligns or conflicts with the Social Sentiment and Technicals.
3. **Actionable Recommendation**: Tell the user what to do (e.g., STRONG BUY, HOLD, SELL, HEDGE). Justify it based on the data points above.

Keep the tone professional, sharp, and data-driven. Do not generate fake data. If there are conflicts (e.g. ML says Up but Social says Down), explain the nuance and suggest a risk-managed approach.
"""
    return prompt

def run_agent(asset: str):
    log.info(f"Starting Market Agent for {asset}...")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        log.error("GEMINI_API_KEY is missing from your .env file!")
        log.error("Please add it: GEMINI_API_KEY=your_key_here")
        sys.exit(1)
        
    # Using gemini-2.0-flash as a fast and highly capable model
    gemini_client = genai.Client(api_key=gemini_key)

    mongo_client = get_mongo_client()
    
    log.info("Fetching latest pipeline outputs from MongoDB...")
    feat_doc = fetch_latest_doc(mongo_client, "features", asset)
    soc_doc = fetch_latest_doc(mongo_client, "social_anomaly", asset)
    pred_doc = fetch_latest_doc(mongo_client, "predictions", asset)
    
    if not feat_doc and not pred_doc:
        log.error(f"No recent data found for {asset} in MongoDB.")
        sys.exit(1)
        
    prompt = format_system_prompt(asset, feat_doc, soc_doc, pred_doc)
    
    log.info("Querying Gemini 2.5 Flash for market analysis...")
    
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
    def _call_gemini():
        return gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    try:
        response = _call_gemini()
        log.info("Analysis Complete. Report follows:\n")
        print("====================================================================")
        print(response.text)
        print("====================================================================")
    except Exception as e:
        log.error(f"Error querying Gemini API after retries: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventOracle LLM Market Agent")
    parser.add_argument("--asset", type=str, choices=KNOWN_ASSETS, help="Asset ticker to analyze")
    args = parser.parse_args()
    
    asset = args.asset
    if not asset:
        asset = input(f"Please enter the asset to analyze (Choices: {', '.join(KNOWN_ASSETS)}): ").strip().upper()
        if asset not in KNOWN_ASSETS:
            print(f"Error: Invalid asset. Must be one of {KNOWN_ASSETS}")
            sys.exit(1)
            
    run_agent(asset)
