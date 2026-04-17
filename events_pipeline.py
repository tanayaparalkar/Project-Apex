"""
events_pipeline_merged.py
EventOracle – Economic Events Pipeline (Merged)
 
Collects macroeconomic and policy events from multiple sources,
engineers surprise/impact features, and upserts into MongoDB Atlas.
 
Sources (in priority order):
  1. FRED API  – US CPI, Fed Funds Rate (free, reliable)
  2. RBI       – MPC decisions scraped from RBI press releases
  3. data.gov.in – India CPI, GDP, IIP (open government data)
  4. Static fallback – full verified 2024–2025 dataset (always runs)
 
Usage:
    python events_pipeline_merged.py                    # Run all event types
    python events_pipeline_merged.py --type CPI         # Single event type
    python events_pipeline_merged.py --type FED_RATE    # Fed decisions only
    python events_pipeline_merged.py --fallback         # Force static fallback only
 
Scheduling (cron – daily at 8:00 AM IST / 2:30 AM UTC):
    30 2 * * * /usr/bin/python3 /path/to/events_pipeline_merged.py >> /var/log/eventoracle_events.log 2>&1
 
Dependencies:
    pip install requests pandas pymongo python-dotenv beautifulsoup4 lxml
"""
 
import os
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional
 
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
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
log = logging.getLogger("eventoracle.events")
 
 
# ╔══════════════════════════════════════════════════════════════════╗
# ║                  ★  CREDENTIALS – FILL THESE IN  ★              ║
# ║                                                                  ║
# ║  Option A (recommended): put them in a .env file next to this   ║
# ║  script and they'll be loaded automatically by python-dotenv.   ║
# ║                                                                  ║
# ║  Option B: hard-code them directly in the variables below.      ║
# ║  (Only do this on a private/local machine, never commit to git) ║
# ╚══════════════════════════════════════════════════════════════════╝
 
# ── REQUIRED ──────────────────────────────────────────────────────
#   Your MongoDB Atlas connection string.
#   Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
#   Find it in Atlas → your cluster → Connect → Drivers
MONGO_URI = os.getenv("MONGO_URI")
 
# ── OPTIONAL but STRONGLY RECOMMENDED ─────────────────────────────
#   FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
#   Without this, FRED calls use a public demo key that rate-limits quickly.
FRED_API_KEY = os.getenv("FRED_API_KEY")
 
# ── OPTIONAL ──────────────────────────────────────────────────────
#   data.gov.in API key (free): https://data.gov.in/user/register
#   The default key below is the public demo key; replace with your own.
DATA_GOV_KEY = os.getenv("DATA_GOV_KEY")
 
# ── .env file template (save as ".env" in the same folder) ────────
#   MONGO_URI=mongodb+srv://youruser:yourpassword@yourcluster.mongodb.net/
#   FRED_API_KEY=abcdef1234567890abcdef1234567890
#   DATA_GOV_KEY=your_data_gov_in_key_here
# ──────────────────────────────────────────────────────────────────
 
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
 
DB_NAME     = "eventoracle"   # ← change if your MongoDB database has a different name
COLLECTION  = "events"        # ← change if you want a different collection name
MAX_RETRIES = 3
RETRY_DELAY = 5               # seconds between retries
BATCH_SIZE  = 200             # MongoDB bulk-write batch size
 
# FRED series IDs (no changes needed)
FRED_SERIES = {
    "US_CPI":   "CPIAUCSL",   # US CPI All Urban Consumers (monthly)
    "FED_RATE": "FEDFUNDS",   # Effective Federal Funds Rate (monthly)
}
 
# RBI MPC press release listing page
RBI_MPC_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
 
# data.gov.in resource IDs (India open data)
DATA_GOV_RESOURCES = {
    "INDIA_CPI": "9ef84268-d588-465a-a308-a864a43d0070",
    "INDIA_IIP": "b9d08ab8-0f5d-4d07-8f97-3d33c79ddd25",
}
DATA_GOV_BASE = "https://api.data.gov.in/resource"
 
 
# ─────────────────────────────────────────────
# STATIC FALLBACK DATA
# Full verified dataset: Jan 2024 – Dec 2025
#
# Sources:
#   India CPI  → MoSPI press releases (pib.gov.in / mospi.gov.in)
#   India IIP  → MoSPI / NSO press releases (pib.gov.in)
#   India GDP  → MoSPI quarterly GDP press releases (pib.gov.in)
#   RBI Repo   → RBI MPC decisions (rbi.org.in)
#   US CPI     → BLS press releases (bls.gov)
#   Fed Rate   → FOMC meeting statements (federalreserve.gov)
#
# Notes:
#   - Oct/Nov 2025 US CPI: BLS suspended release (govt shutdown);
#     value is Cleveland Fed nowcast (2.7%).
#   - RBI cut 5× in 2025 (Feb, Apr, Jun, Aug, Dec) → total 125 bps → 5.25%.
#   - Fed held all of 2025 at 4.25–4.50%; one cut in Dec 2025 (−25 bps).
# ─────────────────────────────────────────────
 
STATIC_EVENTS = [
 
    # ══════════════════════════════════════════════════════════════
    # INDIA CPI  (MoSPI, base 2012=100, YoY %)
    # Release: 12th of the month following the reference month
    # ══════════════════════════════════════════════════════════════
 
    # ── 2024 ──
    {"event_name": "CPI", "country": "India", "timestamp": "2024-01-12T10:30:00Z", "actual": 5.69, "consensus": 5.85, "previous": 5.55},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-02-12T10:30:00Z", "actual": 5.09, "consensus": 5.30, "previous": 5.69},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-03-12T10:30:00Z", "actual": 4.85, "consensus": 5.05, "previous": 5.09},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-04-12T10:30:00Z", "actual": 4.83, "consensus": 4.90, "previous": 4.85},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-05-13T10:30:00Z", "actual": 4.75, "consensus": 4.80, "previous": 4.83},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-06-12T10:30:00Z", "actual": 5.08, "consensus": 4.90, "previous": 4.75},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-07-12T10:30:00Z", "actual": 3.54, "consensus": 3.80, "previous": 5.08},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-08-12T10:30:00Z", "actual": 3.65, "consensus": 3.70, "previous": 3.54},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-09-12T10:30:00Z", "actual": 5.49, "consensus": 4.90, "previous": 3.65},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-10-14T10:30:00Z", "actual": 6.21, "consensus": 5.80, "previous": 5.49},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-11-12T10:30:00Z", "actual": 5.48, "consensus": 5.70, "previous": 6.21},
    {"event_name": "CPI", "country": "India", "timestamp": "2024-12-13T10:30:00Z", "actual": 5.22, "consensus": 5.30, "previous": 5.48},
 
    # ── 2025 ──
    {"event_name": "CPI", "country": "India", "timestamp": "2025-01-13T10:30:00Z", "actual": 4.31, "consensus": 4.60, "previous": 5.22},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-02-12T10:30:00Z", "actual": 3.61, "consensus": 4.10, "previous": 4.31},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-03-14T10:30:00Z", "actual": 3.34, "consensus": 3.80, "previous": 3.61},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-04-14T10:30:00Z", "actual": 3.16, "consensus": 3.50, "previous": 3.34},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-05-13T10:30:00Z", "actual": 2.82, "consensus": 3.10, "previous": 3.16},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-06-12T10:30:00Z", "actual": 2.42, "consensus": 2.80, "previous": 2.82},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-07-14T10:30:00Z", "actual": 1.60, "consensus": 2.20, "previous": 2.42},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-08-12T10:30:00Z", "actual": 2.07, "consensus": 2.00, "previous": 1.60},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-09-12T10:30:00Z", "actual": 1.54, "consensus": 2.00, "previous": 2.07},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-10-14T10:30:00Z", "actual": 0.25, "consensus": 1.20, "previous": 1.54},  # GST rationalisation impact
    {"event_name": "CPI", "country": "India", "timestamp": "2025-11-12T10:30:00Z", "actual": 0.71, "consensus": 0.80, "previous": 0.25},
    {"event_name": "CPI", "country": "India", "timestamp": "2025-12-12T10:30:00Z", "actual": 1.33, "consensus": 0.90, "previous": 0.71},
 
    # ══════════════════════════════════════════════════════════════
    # US CPI  (BLS, YoY %, CPI-U All Items)
    # Release: ~10th–15th of the following month at 8:30 AM ET
    # ══════════════════════════════════════════════════════════════
 
    # ── 2024 ──
    {"event_name": "CPI", "country": "US", "timestamp": "2024-01-11T13:30:00Z", "actual": 3.4,  "consensus": 3.2,  "previous": 3.1},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-02-13T13:30:00Z", "actual": 3.1,  "consensus": 2.9,  "previous": 3.4},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-03-12T12:30:00Z", "actual": 3.2,  "consensus": 3.1,  "previous": 3.1},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-04-10T12:30:00Z", "actual": 3.5,  "consensus": 3.4,  "previous": 3.2},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-05-15T12:30:00Z", "actual": 3.4,  "consensus": 3.4,  "previous": 3.5},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-06-12T12:30:00Z", "actual": 3.0,  "consensus": 3.1,  "previous": 3.4},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-07-11T12:30:00Z", "actual": 2.9,  "consensus": 3.0,  "previous": 3.0},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-08-14T12:30:00Z", "actual": 2.9,  "consensus": 2.9,  "previous": 2.9},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-09-11T12:30:00Z", "actual": 2.5,  "consensus": 2.6,  "previous": 2.9},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-10-10T12:30:00Z", "actual": 2.4,  "consensus": 2.3,  "previous": 2.5},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-11-13T13:30:00Z", "actual": 2.6,  "consensus": 2.6,  "previous": 2.4},
    {"event_name": "CPI", "country": "US", "timestamp": "2024-12-11T13:30:00Z", "actual": 2.7,  "consensus": 2.7,  "previous": 2.6},
 
    # ── 2025 ──
    {"event_name": "CPI", "country": "US", "timestamp": "2025-01-15T13:30:00Z", "actual": 2.9,  "consensus": 2.9,  "previous": 2.7},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-02-12T13:30:00Z", "actual": 3.0,  "consensus": 2.9,  "previous": 2.9},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-03-12T12:30:00Z", "actual": 2.8,  "consensus": 2.9,  "previous": 3.0},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-04-10T12:30:00Z", "actual": 2.4,  "consensus": 2.6,  "previous": 2.8},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-05-13T12:30:00Z", "actual": 2.3,  "consensus": 2.4,  "previous": 2.4},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-06-11T12:30:00Z", "actual": 2.7,  "consensus": 2.5,  "previous": 2.3},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-07-15T12:30:00Z", "actual": 2.9,  "consensus": 2.8,  "previous": 2.7},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-08-12T12:30:00Z", "actual": 2.9,  "consensus": 2.9,  "previous": 2.9},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-09-11T12:30:00Z", "actual": 2.4,  "consensus": 2.5,  "previous": 2.9},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-10-15T12:30:00Z", "actual": 2.7,  "consensus": 2.7,  "previous": 2.4},  # Cleveland Fed nowcast
    {"event_name": "CPI", "country": "US", "timestamp": "2025-11-13T13:30:00Z", "actual": 2.7,  "consensus": 2.7,  "previous": 2.7},
    {"event_name": "CPI", "country": "US", "timestamp": "2025-12-10T13:30:00Z", "actual": 2.7,  "consensus": 2.7,  "previous": 2.7},
 
    # ══════════════════════════════════════════════════════════════
    # US FED RATE  (FOMC upper bound, %)
    # 2024: held → cut 3× (Sep −50bps, Nov −25bps, Dec −25bps)
    # 2025: held all year → one cut Dec 2025 (−25bps → 4.25%)
    # ══════════════════════════════════════════════════════════════
 
    # ── 2024 ──
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-01-31T19:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-03-20T18:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-05-01T18:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-06-12T18:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-07-31T18:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-09-18T18:00:00Z", "actual": 5.00, "consensus": 5.25, "previous": 5.50},  # −50 bps surprise
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-11-07T19:00:00Z", "actual": 4.75, "consensus": 4.75, "previous": 5.00},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2024-12-18T19:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.75},
 
    # ── 2025 ──
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-01-29T19:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-03-19T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-05-07T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-06-18T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-07-30T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-09-17T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-10-29T18:00:00Z", "actual": 4.50, "consensus": 4.50, "previous": 4.50},
    {"event_name": "FED_RATE", "country": "US", "timestamp": "2025-12-10T19:00:00Z", "actual": 4.25, "consensus": 4.25, "previous": 4.50},  # −25 bps
 
    # ══════════════════════════════════════════════════════════════
    # RBI REPO RATE  (%)
    # 2024: held at 6.50% all 6 meetings
    # 2025: cut −25 bps each in Feb, Apr, Jun, Aug, Dec → 5.25%
    # ══════════════════════════════════════════════════════════════
 
    # ── 2024 ──
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-02-08T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-04-05T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-06-07T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-08-08T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-10-09T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2024-12-06T10:00:00Z", "actual": 6.50, "consensus": 6.50, "previous": 6.50},
 
    # ── 2025 ──
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-02-07T10:00:00Z", "actual": 6.25, "consensus": 6.25, "previous": 6.50},  # −25 bps
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-04-09T10:00:00Z", "actual": 6.00, "consensus": 6.00, "previous": 6.25},  # −25 bps
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-06-06T10:00:00Z", "actual": 5.75, "consensus": 5.75, "previous": 6.00},  # −25 bps
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-08-06T10:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.75},  # −25 bps
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-10-01T10:00:00Z", "actual": 5.50, "consensus": 5.50, "previous": 5.50},  # hold
    {"event_name": "RBI_REPO", "country": "India", "timestamp": "2025-12-05T10:00:00Z", "actual": 5.25, "consensus": 5.25, "previous": 5.50},  # −25 bps
 
    # ══════════════════════════════════════════════════════════════
    # INDIA GDP  (YoY real GDP growth %, quarterly)
    # Timestamps = MoSPI press note release date
    # ══════════════════════════════════════════════════════════════
 
    {"event_name": "GDP", "country": "India", "timestamp": "2024-05-31T00:00:00Z", "actual": 7.8,  "consensus": 7.0,  "previous": 8.4},   # Q4 FY24
    {"event_name": "GDP", "country": "India", "timestamp": "2024-08-30T00:00:00Z", "actual": 6.7,  "consensus": 7.0,  "previous": 7.8},   # Q1 FY25
    {"event_name": "GDP", "country": "India", "timestamp": "2024-11-29T00:00:00Z", "actual": 5.4,  "consensus": 6.5,  "previous": 6.7},   # Q2 FY25 – shock miss
    {"event_name": "GDP", "country": "India", "timestamp": "2025-02-28T00:00:00Z", "actual": 6.2,  "consensus": 6.4,  "previous": 5.4},   # Q3 FY25
    {"event_name": "GDP", "country": "India", "timestamp": "2025-05-30T00:00:00Z", "actual": 7.4,  "consensus": 7.0,  "previous": 6.2},   # Q4 FY25
    {"event_name": "GDP", "country": "India", "timestamp": "2025-08-29T00:00:00Z", "actual": 7.8,  "consensus": 7.5,  "previous": 7.4},   # Q1 FY26
    {"event_name": "GDP", "country": "India", "timestamp": "2025-11-28T00:00:00Z", "actual": 8.2,  "consensus": 7.8,  "previous": 7.8},   # Q2 FY26
 
    # ══════════════════════════════════════════════════════════════
    # INDIA IIP  (YoY growth %)
    # Release: 12th of each month with ~42-day lag
    # ══════════════════════════════════════════════════════════════
 
    # ── 2024 ──
    {"event_name": "IIP", "country": "India", "timestamp": "2024-01-12T10:30:00Z", "actual": 4.2,  "consensus": 4.0,  "previous": 2.4},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-02-09T10:30:00Z", "actual": 3.8,  "consensus": 3.5,  "previous": 4.2},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-03-12T10:30:00Z", "actual": 3.6,  "consensus": 3.8,  "previous": 3.8},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-04-12T10:30:00Z", "actual": 5.7,  "consensus": 5.0,  "previous": 3.6},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-05-14T10:30:00Z", "actual": 4.9,  "consensus": 4.5,  "previous": 5.7},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-06-12T10:30:00Z", "actual": 5.0,  "consensus": 4.8,  "previous": 4.9},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-07-12T10:30:00Z", "actual": 5.9,  "consensus": 4.9,  "previous": 5.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-08-12T10:30:00Z", "actual": 4.2,  "consensus": 5.0,  "previous": 5.9},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-09-13T10:30:00Z", "actual": 4.8,  "consensus": 4.5,  "previous": 4.2},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-10-11T10:30:00Z", "actual": 4.5,  "consensus": 4.5,  "previous": 4.8},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-11-12T10:30:00Z", "actual": 3.7,  "consensus": 4.0,  "previous": 4.5},
    {"event_name": "IIP", "country": "India", "timestamp": "2024-12-13T10:30:00Z", "actual": 3.5,  "consensus": 3.5,  "previous": 3.7},
 
    # ── 2025 ──
    {"event_name": "IIP", "country": "India", "timestamp": "2025-01-14T10:30:00Z", "actual": 5.2,  "consensus": 4.5,  "previous": 3.5},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-02-12T10:30:00Z", "actual": 3.2,  "consensus": 3.5,  "previous": 5.2},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-03-14T10:30:00Z", "actual": 5.0,  "consensus": 4.5,  "previous": 3.2},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-04-11T10:30:00Z", "actual": 2.9,  "consensus": 3.2,  "previous": 5.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-05-12T10:30:00Z", "actual": 3.0,  "consensus": 3.3,  "previous": 2.9},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-06-13T10:30:00Z", "actual": 1.5,  "consensus": 2.5,  "previous": 3.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-07-11T10:30:00Z", "actual": 1.5,  "consensus": 2.0,  "previous": 1.5},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-08-12T10:30:00Z", "actual": 4.3,  "consensus": 2.5,  "previous": 1.5},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-09-12T10:30:00Z", "actual": 4.0,  "consensus": 3.8,  "previous": 4.3},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-10-13T10:30:00Z", "actual": 4.0,  "consensus": 4.0,  "previous": 4.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-11-12T10:30:00Z", "actual": 0.4,  "consensus": 2.5,  "previous": 4.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-12-01T10:30:00Z", "actual": 0.4,  "consensus": 2.0,  "previous": 4.0},
    {"event_name": "IIP", "country": "India", "timestamp": "2025-12-29T10:30:00Z", "actual": 6.7,  "consensus": 3.0,  "previous": 0.4},
]
 
# Map event names to category
CATEGORY_MAP = {
    "CPI":      "macro",
    "GDP":      "macro",
    "IIP":      "macro",
    "FED_RATE": "policy",
    "RBI_REPO": "policy",
}
 
 
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
 
def _get(url: str, params: dict = None, timeout: int = 15) -> Optional[requests.Response]:
    """GET with retry logic. Returns Response or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"  HTTP attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    log.error(f"  All {MAX_RETRIES} attempts failed for: {url}")
    return None
 
 
def _to_utc(ts) -> Optional[datetime]:
    """Coerce various timestamp formats to UTC-aware datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(ts.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    log.debug(f"  Could not parse timestamp: '{ts}'")
    return None
 
 
# ─────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────
 
def fetch_fred(series_id: str, event_name: str) -> list[dict]:
    """Fetch a FRED time series and return raw event dicts."""
    log.info(f"  [FRED] Fetching '{series_id}' for '{event_name}'...")
    params = {
        "series_id":         series_id,
        "file_type":         "json",
        "observation_start": "2022-01-01",
        "sort_order":        "asc",
        "api_key":           FRED_API_KEY if FRED_API_KEY != "YOUR_FRED_API_KEY_HERE" else "abcdefghijklmnopqrstuvwxyz123456",
    }
    resp = _get("https://api.stlouisfed.org/fred/series/observations", params=params)
    if resp is None:
        return []
    try:
        observations = resp.json().get("observations", [])
    except Exception as e:
        log.error(f"  [FRED] JSON parse error: {e}")
        return []
 
    records = []
    for obs in observations:
        raw_val = obs.get("value", ".")
        if raw_val == ".":
            continue
        try:
            actual = float(raw_val)
        except ValueError:
            continue
        ts = _to_utc(obs.get("date"))
        if ts is None:
            continue
        records.append({
            "event_name": event_name,
            "country":    "US",
            "timestamp":  ts,
            "actual":     actual,
            "consensus":  None,
            "previous":   None,
        })
    log.info(f"  [FRED] Retrieved {len(records)} observations.")
    return records
 
 
def fetch_rbi_mpc() -> list[dict]:
    """Scrape RBI press releases for MPC repo rate decisions."""
    log.info("  [RBI] Scraping MPC decisions...")
    params = {"fn": "4", "regid": "21"}
    resp = _get(RBI_MPC_URL, params=params)
    if resp is None:
        log.warning("  [RBI] Scraping failed. Skipping live RBI data.")
        return []
    records = []
    try:
        soup = BeautifulSoup(resp.text, "lxml")
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            title = cells[1].get_text(strip=True).lower()
            if "monetary policy" not in title and "repo rate" not in title:
                continue
            ts = _to_utc(cells[0].get_text(strip=True))
            if ts is None:
                continue
            records.append({
                "event_name":  "RBI_REPO",
                "country":     "India",
                "timestamp":   ts,
                "actual":      None,
                "consensus":   None,
                "previous":    None,
                "_source_url": cells[1].find("a", href=True)["href"] if cells[1].find("a") else None,
            })
        log.info(f"  [RBI] Found {len(records)} MPC press releases.")
    except Exception as e:
        log.error(f"  [RBI] Parsing error: {e}")
    return records
 
 
def fetch_data_gov(resource_id: str, event_name: str, country: str = "India") -> list[dict]:
    """Fetch India macro data from data.gov.in open API."""
    log.info(f"  [data.gov.in] Fetching '{event_name}' (resource: {resource_id})...")
    params = {"api-key": DATA_GOV_KEY, "format": "json", "limit": "500"}
    resp = _get(f"{DATA_GOV_BASE}/{resource_id}", params=params)
    if resp is None:
        return []
    try:
        rows = resp.json().get("records", [])
    except Exception as e:
        log.error(f"  [data.gov.in] JSON parse error: {e}")
        return []
 
    records = []
    for row in rows:
        row_lower = {k.lower().strip(): v for k, v in row.items()}
        ts = None
        for col in ["date", "month", "period", "year", "release_date"]:
            if col in row_lower and row_lower[col]:
                ts = _to_utc(str(row_lower[col]))
                if ts:
                    break
        if ts is None:
            continue
        actual = None
        for col in ["value", "actual", "index", "rate", "percent", "growth"]:
            if col in row_lower:
                try:
                    actual = float(str(row_lower[col]).replace(",", ""))
                    break
                except (ValueError, TypeError):
                    continue
        if actual is None:
            continue
        records.append({
            "event_name": event_name,
            "country":    country,
            "timestamp":  ts,
            "actual":     actual,
            "consensus":  None,
            "previous":   None,
        })
    log.info(f"  [data.gov.in] Retrieved {len(records)} records.")
    return records
 
 
def fetch_static_fallback(event_type: str = None) -> list[dict]:
    """Return the full verified static dataset, optionally filtered by event type."""
    log.info("  [STATIC] Loading verified fallback data...")
    records = []
    for ev in STATIC_EVENTS:
        if event_type and ev["event_name"] != event_type:
            continue
        rec = dict(ev)
        rec["timestamp"] = _to_utc(rec["timestamp"])
        if rec["timestamp"] is None:
            continue
        records.append(rec)
    log.info(f"  [STATIC] Loaded {len(records)} records.")
    return records
 
 
# ─────────────────────────────────────────────
# 2. CLEANING & NORMALIZATION
# ─────────────────────────────────────────────
 
def clean_events(records: list[dict]) -> pd.DataFrame:
    """
    Normalize raw records:
    - Enforce schema columns
    - Deduplicate on (event_name, country, timestamp)
    - Coerce numeric types
    - Back-fill 'previous' by lag within each group
    """
    if not records:
        return pd.DataFrame()
 
    df = pd.DataFrame(records)
    for col in ["event_name", "country", "timestamp", "actual", "consensus", "previous"]:
        if col not in df.columns:
            df[col] = None
 
    df["actual"]    = pd.to_numeric(df["actual"],    errors="coerce")
    df["consensus"] = pd.to_numeric(df["consensus"], errors="coerce")
    df["previous"]  = pd.to_numeric(df["previous"],  errors="coerce")
    df["timestamp"] = df["timestamp"].apply(
        lambda t: t if isinstance(t, datetime) else _to_utc(str(t)) if t else None
    )
 
    before = len(df)
    df = df.dropna(subset=["timestamp", "actual"])
    df = df.drop_duplicates(subset=["event_name", "country", "timestamp"])
    after = len(df)
    if before != after:
        log.info(f"  Dropped {before - after} null/duplicate rows. Remaining: {after}")
 
    df = df.sort_values(["event_name", "country", "timestamp"]).reset_index(drop=True)
 
    # Back-fill 'previous' from preceding row where missing
    for (ev, ctry), grp in df.groupby(["event_name", "country"]):
        mask = df["event_name"].eq(ev) & df["country"].eq(ctry)
        df.loc[mask, "_lagged"] = df.loc[mask, "actual"].shift(1)
    fill_mask = df["previous"].isna() & df["_lagged"].notna()
    df.loc[fill_mask, "previous"] = df.loc[fill_mask, "_lagged"]
    df = df.drop(columns=["_lagged"], errors="ignore")
 
    df["category"] = df["event_name"].map(CATEGORY_MAP).fillna("macro")
    return df.reset_index(drop=True)
 
 
# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
 
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
    - surprise     = actual - consensus
    - impact_score = z-scored surprise per (event_name, country)
    """
    if df.empty:
        return df
 
    df["surprise"] = df.apply(
        lambda row: round(row["actual"] - row["consensus"], 6)
        if pd.notna(row["actual"]) and pd.notna(row["consensus"])
        else None,
        axis=1,
    )
 
    df["impact_score"] = np.nan
    for (ev, ctry), grp in df.groupby(["event_name", "country"]):
        idx       = grp.index
        surprises = grp["surprise"].dropna()
        if surprises.empty:
            actuals = grp["actual"].dropna()
            std, mean = actuals.std(), actuals.mean()
            if std and std > 0:
                df.loc[idx, "impact_score"] = ((grp["actual"] - mean) / std).round(6)
            continue
        mean, std = surprises.mean(), surprises.std()
        if pd.isna(std) or std == 0:
            df.loc[surprises.index, "impact_score"] = np.sign(surprises).round(6)
        else:
            df.loc[surprises.index, "impact_score"] = ((surprises - mean) / std).round(6)
 
    for col in ["actual", "consensus", "previous", "surprise", "impact_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(6)
 
    log.info(
        f"  Features done. surprise: {df['surprise'].notna().sum()}/{len(df)} | "
        f"impact_score: {df['impact_score'].notna().sum()}/{len(df)}"
    )
    return df
 
 
# ─────────────────────────────────────────────
# 4. DATABASE STORAGE
# ─────────────────────────────────────────────
 
def get_mongo_collection():
    """
    Connect to MongoDB Atlas and return the events collection.
 
    ★ CREDENTIAL CHECK ★
    Make sure MONGO_URI at the top of this file is set —
    either via .env or by replacing the placeholder string.
    """
    if not MONGO_URI or MONGO_URI == "YOUR_MONGODB_CONNECTION_STRING_HERE":
        raise ValueError(
            "MONGO_URI is not configured.\n"
            "Set it in your .env file or replace 'YOUR_MONGODB_CONNECTION_STRING_HERE' "
            "at the top of this script."
        )
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")   # fast connectivity check
    col = client[DB_NAME][COLLECTION]
    col.create_index(
        [("event_name", 1), ("country", 1), ("timestamp", 1)],
        unique=True,
        background=True,
        name="event_dedup_idx",
    )
    return col
 
 
def upload_to_mongo(df: pd.DataFrame, collection) -> int:
    """Upsert records; returns count of newly inserted documents."""
    if df.empty:
        log.warning("  Nothing to upload — dataframe is empty.")
        return 0
 
    output_cols = [c for c in [
        "event_name", "country", "timestamp",
        "actual", "consensus", "previous",
        "surprise", "impact_score", "category",
    ] if c in df.columns]
 
    records = df[output_cols].to_dict("records")
 
    # Sanitize types for MongoDB
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, np.integer):
                rec[k] = int(v)
            elif isinstance(v, np.floating):
                rec[k] = float(v) if not np.isnan(v) else None
 
    ops = [
        UpdateOne(
            {"event_name": r["event_name"], "country": r["country"], "timestamp": r["timestamp"]},
            {"$setOnInsert": r},
            upsert=True,
        )
        for r in records
    ]
 
    total_upserted = 0
    for i in range(0, len(ops), BATCH_SIZE):
        try:
            result = collection.bulk_write(ops[i : i + BATCH_SIZE], ordered=False)
            total_upserted += result.upserted_count
        except BulkWriteError as bwe:
            total_upserted += bwe.details.get("nUpserted", 0)
            log.debug(f"  BulkWriteError (expected on reruns): {bwe.details}")
 
    log.info(f"  ✅ Upserted {total_upserted} new records into '{COLLECTION}'.")
    return total_upserted
 
 
# ─────────────────────────────────────────────
# 5. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────
 
def run_pipeline(event_type: str = None, force_fallback: bool = False):
    """Full pipeline: fetch → fallback → merge → clean → features → upsert."""
    log.info("=" * 60)
    log.info("EventOracle – Economic Events Pipeline Starting")
    log.info(f"Mode: {'static fallback only' if force_fallback else 'live APIs + static fallback'} | "
             f"Filter: {event_type or 'all events'}")
    log.info("=" * 60)
 
    try:
        collection = get_mongo_collection()
        log.info("✅ Connected to MongoDB Atlas.")
    except Exception as e:
        log.error(f"❌ MongoDB connection failed: {e}")
        return
 
    all_records: list[dict] = []
 
    if not force_fallback:
        if event_type in (None, "CPI"):
            log.info("\n── US CPI (FRED) ──")
            try:
                all_records.extend(fetch_fred(FRED_SERIES["US_CPI"], "CPI"))
            except Exception as e:
                log.error(f"  [FRED CPI] {e}")
 
        if event_type in (None, "FED_RATE"):
            log.info("\n── Fed Funds Rate (FRED) ──")
            try:
                all_records.extend(fetch_fred(FRED_SERIES["FED_RATE"], "FED_RATE"))
            except Exception as e:
                log.error(f"  [FRED FED_RATE] {e}")
 
        if event_type in (None, "CPI"):
            log.info("\n── India CPI (data.gov.in) ──")
            try:
                all_records.extend(fetch_data_gov(DATA_GOV_RESOURCES["INDIA_CPI"], "CPI", "India"))
            except Exception as e:
                log.error(f"  [data.gov.in CPI] {e}")
 
        if event_type in (None, "IIP"):
            log.info("\n── India IIP (data.gov.in) ──")
            try:
                all_records.extend(fetch_data_gov(DATA_GOV_RESOURCES["INDIA_IIP"], "IIP", "India"))
            except Exception as e:
                log.error(f"  [data.gov.in IIP] {e}")
 
        if event_type in (None, "RBI_REPO"):
            log.info("\n── RBI MPC Repo Rate (scrape) ──")
            try:
                all_records.extend(fetch_rbi_mpc())
            except Exception as e:
                log.error(f"  [RBI scrape] {e}")
 
    # Static fallback always runs — it supplements live data
    log.info("\n── Static Verified Fallback (always runs) ──")
    all_records.extend(fetch_static_fallback(event_type))
 
    if not all_records:
        log.error("❌ Pipeline aborted — no records from any source.")
        return
 
    log.info(f"\n── Total raw records: {len(all_records)} ──")
 
    log.info("\n── Cleaning & Normalizing ──")
    df = clean_events(all_records)
    if df.empty:
        log.error("❌ Pipeline aborted — all records dropped during cleaning.")
        return
 
    log.info("\n── Feature Engineering ──")
    df = add_features(df)
 
    log.info("\n── Uploading to MongoDB ──")
    inserted = upload_to_mongo(df, collection)
 
    log.info("\n" + "=" * 60)
    log.info("Pipeline Complete – Summary")
    log.info("=" * 60)
    log.info(f"  Total records processed : {len(df)}")
    log.info(f"  New records inserted    : {inserted}")
    for key, cnt in df.groupby(["event_name", "country"]).size().to_dict().items():
        log.info(f"  {key[0]:<12} ({key[1]:<6}) → {cnt} records")
    log.info("=" * 60)
 
 
# ─────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventOracle Economic Events Pipeline")
    parser.add_argument(
        "--type", default=None,
        choices=["CPI", "GDP", "IIP", "FED_RATE", "RBI_REPO"],
        help="Run for a single event type (default: all)",
    )
    parser.add_argument(
        "--fallback", action="store_true", default=False,
        help="Skip live API calls; load static fallback only",
    )
    args = parser.parse_args()
    run_pipeline(event_type=args.type, force_fallback=args.fallback)