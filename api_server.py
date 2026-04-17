"""
api_server.py
EventOracle – REST API Backend for the Dashboard

Serves data from MongoDB to the React frontend and wraps
pipeline outputs (market_agent, modeling) for live interaction.

Usage:
    python api_server.py
    # → runs on http://localhost:5000

Endpoints:
    GET  /api/assets              → list of tracked assets
    GET  /api/prices/<asset>      → OHLCV + technical data
    GET  /api/features/<asset>    → latest feature row
    GET  /api/news                → recent news articles
    GET  /api/news/<asset>        → news filtered by asset keywords
    GET  /api/predictions/<asset> → model predictions
    GET  /api/social/<asset>      → social anomaly data
    POST /api/agent/<asset>       → run market agent (Gemini)
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.api")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = "eventoracle"
IST       = timezone(timedelta(hours=5, minutes=30))

# Path to the React build output (npm run build → frontend/dist/)
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
)
CORS(app)

KNOWN_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]

# Asset → keywords for news filtering
ASSET_KEYWORDS = {
    "NIFTY":      ["nifty", "sensex", "nse", "bse"],
    "BANKNIFTY":  ["bank nifty", "banknifty", "banking"],
    "NIFTYIT":    ["nifty it", "it sector", "tech stocks"],
    "NIFTYMETAL": ["nifty metal", "metal sector", "steel"],
    "INRUSD":     ["rupee", "inr", "dollar", "usd", "forex", "currency"],
    "BRENT":      ["oil", "brent", "crude", "opec", "petroleum"],
    "GOLD":       ["gold", "bullion", "yellow metal", "xauusd"],
    "SILVER":     ["silver", "xagusd"],
    "PLATINUM":   ["platinum", "pgm"],
    "BITCOIN":    ["bitcoin", "btc", "crypto", "cryptocurrency"],
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

_client: Optional[MongoClient] = None

def get_db():
    global _client
    if _client is None:
        if not MONGO_URI:
            raise ValueError("MONGO_URI not set")
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        _client.admin.command("ping")
        log.info("✅ Connected to MongoDB Atlas.")
    return _client[DB_NAME]


def serialize(docs):
    """Convert MongoDB docs to JSON-safe dicts."""
    result = []
    for doc in docs:
        row = {}
        for k, v in doc.items():
            if k == "_id":
                continue
            if isinstance(v, datetime):
                row[k] = v.isoformat()
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
            else:
                row[k] = v
        result.append(row)
    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/api/assets", methods=["GET"])
def get_assets():
    """Return list of tracked assets with metadata."""
    asset_info = []
    db = get_db()
    for asset in KNOWN_ASSETS:
        # Get latest price for each asset
        latest = db["prices"].find_one(
            {"asset": asset},
            sort=[("timestamp", -1)],
            projection={"_id": 0, "close": 1, "returns": 1, "timestamp": 1}
        )
        info = {"name": asset, "close": None, "returns": None, "timestamp": None}
        if latest:
            info["close"] = latest.get("close")
            info["returns"] = latest.get("returns")
            ts = latest.get("timestamp")
            info["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else str(ts) if ts else None
        asset_info.append(info)
    return jsonify(asset_info)


@app.route("/api/prices/<asset>", methods=["GET"])
def get_prices(asset: str):
    """Return OHLCV + technical data for asset. Optional ?limit=N, ?days=N."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()
    limit = request.args.get("limit", 500, type=int)
    days = request.args.get("days", None, type=int)

    query = {"asset": asset}
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query["timestamp"] = {"$gte": cutoff}

    cursor = db["prices"].find(
        query,
        {"_id": 0}
    ).sort("timestamp", 1).limit(limit)

    docs = list(cursor)
    return jsonify(serialize(docs))


@app.route("/api/features/<asset>", methods=["GET"])
def get_features(asset: str):
    """Return latest feature row for an asset."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()
    limit = request.args.get("limit", 30, type=int)

    cursor = db["features"].find(
        {"asset": asset},
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit)

    docs = list(cursor)
    docs.reverse()  # chronological order
    return jsonify(serialize(docs))


@app.route("/api/news", methods=["GET"])
def get_all_news():
    """Return recent news articles. Optional ?limit=N."""
    db = get_db()
    limit = request.args.get("limit", 50, type=int)

    cursor = db["news"].find(
        {},
        {"_id": 0, "headline": 1, "source": 1, "timestamp": 1,
         "sentiment_score": 1, "sentiment_label": 1, "event_type": 1, "link": 1}
    ).sort("timestamp", -1).limit(limit)

    docs = list(cursor)
    return jsonify(serialize(docs))


@app.route("/api/news/<asset>", methods=["GET"])
def get_news_by_asset(asset: str):
    """Return news filtered by asset keywords."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()
    limit = request.args.get("limit", 30, type=int)
    keywords = ASSET_KEYWORDS.get(asset, [asset.lower()])

    # Build regex OR query on headline
    regex_pattern = "|".join(keywords)
    cursor = db["news"].find(
        {"headline": {"$regex": regex_pattern, "$options": "i"}},
        {"_id": 0, "headline": 1, "source": 1, "timestamp": 1,
         "sentiment_score": 1, "sentiment_label": 1, "event_type": 1, "link": 1}
    ).sort("timestamp", -1).limit(limit)

    docs = list(cursor)

    # If keyword-filtered results are sparse, also return latest general news
    if len(docs) < 5:
        general = db["news"].find(
            {},
            {"_id": 0, "headline": 1, "source": 1, "timestamp": 1,
             "sentiment_score": 1, "sentiment_label": 1, "event_type": 1, "link": 1}
        ).sort("timestamp", -1).limit(20)
        existing_headlines = {d.get("headline") for d in docs}
        for g in general:
            if g.get("headline") not in existing_headlines:
                docs.append(g)
            if len(docs) >= limit:
                break

    return jsonify(serialize(docs))


@app.route("/api/predictions/<asset>", methods=["GET"])
def get_predictions(asset: str):
    """Return model predictions for an asset."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()
    limit = request.args.get("limit", 30, type=int)

    cursor = db["predictions"].find(
        {"asset": asset},
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit)

    docs = list(cursor)
    docs.reverse()
    return jsonify(serialize(docs))


@app.route("/api/social/<asset>", methods=["GET"])
def get_social(asset: str):
    """Return social anomaly data for an asset."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()
    limit = request.args.get("limit", 30, type=int)

    cursor = db["social_anomaly"].find(
        {"asset": asset},
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit)

    docs = list(cursor)
    docs.reverse()
    return jsonify(serialize(docs))


@app.route("/api/agent/<asset>", methods=["POST"])
def run_market_agent(asset: str):
    """
    Run the market agent for a given asset.
    Fetches latest data from MongoDB and calls Gemini.
    """
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return jsonify({
            "error": "GEMINI_API_KEY not configured",
            "report": "⚠️ **Gemini API key not set.** Add `GEMINI_API_KEY=your_key` to your `.env` file to enable the AI Market Agent."
        }), 200

    try:
        from google import genai
        db = get_db()

        # Fetch latest docs
        feat_doc = db["features"].find_one(
            {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
        ) or {}
        soc_doc = db["social_anomaly"].find_one(
            {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
        ) or {}
        pred_doc = db["predictions"].find_one(
            {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
        ) or {}

        if not feat_doc and not pred_doc:
            return jsonify({"error": f"No data found for {asset}", "report": f"No data available for {asset} in the database."})

        # Build prompt (same as market_agent.py)
        from market_agent import format_system_prompt
        prompt = format_system_prompt(asset, feat_doc, soc_doc, pred_doc)

        gemini_client = genai.Client(api_key=gemini_key)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({"report": response.text, "asset": asset})

    except Exception as e:
        log.error(f"Agent error for {asset}: {e}", exc_info=True)
        return jsonify({"error": str(e), "report": f"Error generating report: {str(e)}"}), 500


@app.route("/api/summary/<asset>", methods=["GET"])
def get_asset_summary(asset: str):
    """Return a combined summary: latest price, prediction, social data."""
    if asset not in KNOWN_ASSETS:
        return jsonify({"error": f"Unknown asset: {asset}"}), 400

    db = get_db()

    price = db["prices"].find_one(
        {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
    ) or {}

    pred = db["predictions"].find_one(
        {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
    ) or {}

    social = db["social_anomaly"].find_one(
        {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
    ) or {}

    feature = db["features"].find_one(
        {"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}
    ) or {}

    summary = {
        "asset": asset,
        "price": {
            "close": price.get("close"),
            "open": price.get("open"),
            "high": price.get("high"),
            "low": price.get("low"),
            "volume": price.get("volume"),
            "returns": price.get("returns"),
            "timestamp": price.get("timestamp", "").isoformat() if isinstance(price.get("timestamp"), datetime) else str(price.get("timestamp", "")),
        },
        "prediction": {
            "direction": pred.get("predicted_direction"),
            "magnitude": pred.get("predicted_magnitude"),
            "confidence": pred.get("confidence_score"),
            "regime": pred.get("regime"),
            "strategy_return": pred.get("strategy_return"),
        },
        "social": {
            "sentiment_score": social.get("social_sentiment_score"),
            "buzz_intensity": social.get("social_buzz_intensity"),
            "anomaly_flag": social.get("composite_anomaly_flag"),
            "market_stress": social.get("market_stress_score"),
        },
        "technicals": {
            "rsi": feature.get("rsi"),
            "atr": feature.get("atr"),
            "rolling_volatility": feature.get("rolling_volatility"),
            "volume_zscore": feature.get("volume_zscore"),
            "avg_sentiment": feature.get("avg_sentiment"),
        },
    }

    return jsonify(summary)


# ─────────────────────────────────────────────
# SPA CATCH-ALL (serves React frontend)
# ─────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """Serve the React SPA. API routes take precedence (registered above)."""
    if path and (FRONTEND_DIR / path).is_file():
        return send_from_directory(str(FRONTEND_DIR), path)
    # For any other route, serve index.html (React handles client-side routing)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting EventOracle API Server...")
    try:
        get_db()
    except Exception as e:
        log.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=(os.getenv("FLASK_ENV") != "production"))
