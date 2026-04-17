"""
modeling.py
EventOracle – Event Labeling, Modeling & Prediction Pipeline (Person 9)
Dependencies: P3-P8 outputs in MongoDB (features + social_anomaly collections)

Reads from : eventoracle.features, eventoracle.social_anomaly
Writes to  : eventoracle.predictions

Usage:
    python modeling.py --train                   # Train, SHAP select, backtest
    python modeling.py --predict                 # Load models, generate predictions
    python modeling.py --train   --asset NIFTY   # Single-asset train
    python modeling.py --predict --asset GOLD    # Single-asset predict

Cron (daily 8 AM IST / 2:30 AM UTC):
    30 2 * * * /usr/bin/python3 /path/to/modeling.py --predict >> /var/log/eventoracle.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import warnings
from datetime import timedelta, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

MONGO_URI          = os.getenv("MONGO_URI")
DB_NAME            = "eventoracle"
FEATURES_COL       = "features"
SOCIAL_COL         = "social_anomaly"
PREDICTIONS_COL    = "predictions"

IST                = timezone(timedelta(hours=5, minutes=30))
BATCH_SIZE         = 500
MODEL_DIR          = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

STAGE1_MODEL_PATH  = MODEL_DIR / "lgbm_stage1_impact.pkl"
STAGE2_MODEL_PATH  = MODEL_DIR / "lgbm_stage2_magnitude.pkl"
FEATURES_PATH      = MODEL_DIR / "selected_features.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "asset_label_encoder.pkl"
CLASS_MAP_PATH     = MODEL_DIR / "class_map.pkl"

TXN_COST           = 0.001    # 0.1% one-way
NEUTRAL_THRESHOLD  = 0.002    # |fwd_return| < 0.2% => neutral
N_SPLITS_CV        = 5
SHAP_DROP_PCT      = 0.02     # drop bottom 2% of SHAP mass
N_REGIMES          = 3
MIN_ROWS_FOR_TRAIN = 30

KNOWN_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]
EQUITY_INDICES = ["NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL"]

# source -> target assets that receive its cross-asset return signal
CROSS_ASSET_MAP = {
    "BRENT":   EQUITY_INDICES,
    "INRUSD":  ["NIFTYIT"],
    "GOLD":    KNOWN_ASSETS,
    "BITCOIN": KNOWN_ASSETS,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eventoracle.modeling")


# ─────────────────────────────────────────────
# MONGO HELPERS
# ─────────────────────────────────────────────

def get_mongo_client() -> MongoClient:
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client


def load_collection(client: MongoClient, collection: str, asset: Optional[str] = None) -> pd.DataFrame:
    col   = client[DB_NAME][collection]
    query = {"asset": asset} if asset else {}
    docs  = list(col.find(query, {"_id": 0}))
    if not docs:
        log.warning(f"'{collection}' returned 0 documents.")
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    return df.sort_values(["asset", "timestamp"]).reset_index(drop=True)


def upsert_predictions(client: MongoClient, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    col = client[DB_NAME][PREDICTIONS_COL]
    col.create_index([("timestamp", 1), ("asset", 1)], unique=True, background=True, name="ts_asset_unique")
    records = df.to_dict("records")
    for rec in records:
        ts = rec.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            rec["timestamp"] = ts.to_pydatetime()
        for k, v in list(rec.items()):
            if isinstance(v, float) and np.isnan(v):      rec[k] = None
            elif isinstance(v, np.integer):                rec[k] = int(v)
            elif isinstance(v, np.floating):               rec[k] = None if np.isnan(v) else float(v)
    ops = [UpdateOne({"timestamp": r["timestamp"], "asset": r["asset"]}, {"$set": r}, upsert=True) for r in records]
    total = 0
    for i in range(0, len(ops), BATCH_SIZE):
        try:
            res = col.bulk_write(ops[i:i+BATCH_SIZE], ordered=False)
            total += res.upserted_count + res.modified_count
        except BulkWriteError as bwe:
            total += bwe.details.get("nUpserted", 0)
    log.info(f"Upserted/updated {total} records into '{PREDICTIONS_COL}'.")
    return total


# ─────────────────────────────────────────────
# 1. EVENT LABELING (strict no-leakage)
# ─────────────────────────────────────────────

def label_events(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").copy()
        if "close" in grp.columns and grp["close"].notna().sum() > 10:
            fwd1 = grp["close"].pct_change().shift(-1)
            fwd2 = grp["close"].pct_change().shift(-2)
        else:
            fwd1 = grp["returns"].shift(-1)
            fwd2 = grp["returns"].shift(-2)
        grp["forward_return_t1"] = fwd1.round(6)
        grp["forward_return_t2"] = fwd2.round(6)
        grp["event_return"]      = ((fwd1 + fwd2) / 2).round(6)
        grp["impact_label"]      = 0
        grp.loc[grp["event_return"] >  NEUTRAL_THRESHOLD, "impact_label"] =  1
        grp.loc[grp["event_return"] < -NEUTRAL_THRESHOLD, "impact_label"] = -1
        grp["magnitude_label"] = grp["event_return"].abs().round(6)
        grp = grp.dropna(subset=["forward_return_t1", "forward_return_t2"])  # hard leakage boundary
        parts.append(grp)
    result = pd.concat(parts, ignore_index=True)
    log.info(f"Event label distribution: {result['impact_label'].value_counts().to_dict()}")
    return result


# ─────────────────────────────────────────────
# 2. FEATURE MERGING
# ─────────────────────────────────────────────

PRICE_FEATURES  = ["returns", "rolling_volatility", "rsi", "atr", "volume_zscore", "open", "high", "low", "close"]
NEWS_FEATURES   = ["avg_sentiment", "news_volume", "news_velocity", "news_spike_flag"]
FLOW_FEATURES   = ["flow_zscore"]
EVENT_FEATURES  = ["surprise", "normalized_surprise"]
SOCIAL_FEATURES = [
    "social_sentiment_score", "social_buzz_intensity", "sentiment_momentum", "sentiment_volatility",
    "anomaly_score", "anomaly_flag", "market_stress_score", "composite_anomaly_flag",
    "flow_anomaly_flag", "price_anomaly_flag",
]
ALL_BASE_FEATURES = PRICE_FEATURES + NEWS_FEATURES + FLOW_FEATURES + EVENT_FEATURES + SOCIAL_FEATURES


def merge_sources(feat_df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
    if social_df.empty:
        log.warning("social_anomaly empty – proceeding without social features.")
        return feat_df
    keep   = ["timestamp", "asset"] + [c for c in SOCIAL_FEATURES if c in social_df.columns]
    slim   = social_df[keep].drop_duplicates(subset=["timestamp", "asset"])
    merged = feat_df.merge(slim, on=["timestamp", "asset"], how="left")
    log.info(f"Merged shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────
# 3. CROSS-ASSET FEATURES
# ─────────────────────────────────────────────

def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    source_series: dict[str, pd.Series] = {}
    for src in CROSS_ASSET_MAP:
        sub = df[df["asset"] == src][["timestamp", "returns"]].drop_duplicates("timestamp")
        if sub.empty:
            log.warning(f"Cross-asset source '{src}' missing – skipping.")
            continue
        source_series[src] = sub.set_index("timestamp")["returns"]
    if not source_series:
        return df
    parts = []
    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").set_index("timestamp").copy()
        for src, targets in CROSS_ASSET_MAP.items():
            if asset not in targets or src not in source_series:
                continue
            base = f"{src.lower()}_xret"
            s    = source_series[src]
            grp[base]           = grp.index.map(s)
            grp[f"{base}_lag1"] = grp[base].shift(1)
            grp[f"{base}_lag2"] = grp[base].shift(2)
        parts.append(grp.reset_index())
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────
# 4. REGIME DETECTION
# ─────────────────────────────────────────────

def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    df        = df.copy()
    vol       = df["rolling_volatility"].fillna(df["rolling_volatility"].median())
    ret       = df["returns"].fillna(0)
    roll_mean = ret.rolling(60, min_periods=10).mean()
    roll_std  = ret.rolling(60, min_periods=10).std().replace(0, np.nan).fillna(1)
    ret_z     = ((ret - roll_mean) / roll_std).fillna(0).clip(-5, 5)
    X         = np.column_stack([vol.values, ret_z.values])
    km        = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10, max_iter=300)
    raw       = km.fit_predict(X)
    order     = np.argsort(km.cluster_centers_[:, 0])
    remap     = {int(old): int(new) for new, old in enumerate(order)}
    df["regime_label"] = [remap[int(l)] for l in raw]
    log.info(f"Regime distribution: {pd.Series(df['regime_label']).value_counts().sort_index().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE MATRIX
# ─────────────────────────────────────────────

SELF_LAG_COLS = ["returns", "rsi", "rolling_volatility", "avg_sentiment", "flow_zscore", "volume_zscore"]


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
    fit_encoder: bool = True,
) -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    df = df.copy()
    if fit_encoder:
        encoder = LabelEncoder()
        encoder.fit(KNOWN_ASSETS)
    df["asset_enc"] = encoder.transform(df["asset"].where(df["asset"].isin(encoder.classes_), KNOWN_ASSETS[0]))
    for col in SELF_LAG_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df.groupby("asset")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("asset")[col].shift(2)
    cross_cols = [c for c in df.columns if "_xret" in c]
    raw_cols   = (
        ["asset_enc", "regime_label"]
        + [c for c in ALL_BASE_FEATURES if c in df.columns]
        + [f"{c}_lag1" for c in SELF_LAG_COLS if f"{c}_lag1" in df.columns]
        + [f"{c}_lag2" for c in SELF_LAG_COLS if f"{c}_lag2" in df.columns]
        + cross_cols
    )
    seen: set[str] = set()
    feature_cols: list[str] = []
    for c in raw_cols:
        if c not in seen:
            seen.add(c)
            feature_cols.append(c)
    df[feature_cols] = df.groupby("asset", group_keys=False)[feature_cols].apply(lambda g: g.ffill(limit=3))
    df[feature_cols] = df[feature_cols].fillna(0)
    log.info(f"Feature matrix: {len(df)} rows x {len(feature_cols)} features.")
    return df, feature_cols, encoder


# ─────────────────────────────────────────────
# 6. LIGHTGBM + SHAP
# ─────────────────────────────────────────────

LGB_CLF_PARAMS = dict(
    objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.04,
    num_leaves=63, min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
)
LGB_REG_PARAMS = dict(
    objective="regression_l1", n_estimators=600, learning_rate=0.04,
    num_leaves=63, min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
)


def _cv_score(params: dict, is_clf: bool, X: pd.DataFrame, y: pd.Series) -> float:
    scores = []
    for tr, val in TimeSeriesSplit(n_splits=N_SPLITS_CV).split(X):
        m = lgb.LGBMClassifier(**params) if is_clf else lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        preds = m.predict(X.iloc[val])
        scores.append(
            f1_score(y.iloc[val], preds, average="macro", zero_division=0)
            if is_clf else -mean_absolute_error(y.iloc[val], preds)
        )
    return float(np.mean(scores))


def _shap_select(model: lgb.LGBMClassifier, X: pd.DataFrame) -> list[str]:
    log.info("Computing SHAP values for feature selection...")
    sv = shap.TreeExplainer(model).shap_values(X)
    mean_abs   = np.mean([np.abs(a).mean(axis=0) for a in sv], axis=0) if isinstance(sv, list) else np.abs(sv).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    total      = importance.sum()
    if total == 0:
        return list(X.columns)
    cum        = importance.cumsum() / total
    selected   = importance[cum <= (1.0 - SHAP_DROP_PCT)].index.tolist()
    if len(selected) < 5:
        selected = importance.head(max(5, len(selected))).index.tolist()
    dropped = [c for c in X.columns if c not in selected]
    log.info(f"SHAP: {len(selected)}/{len(X.columns)} kept, dropped {len(dropped)}: {dropped[:8]}{'...' if len(dropped)>8 else ''}")
    return selected


# ─────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────

def train(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    df = df.dropna(subset=["impact_label", "magnitude_label"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < MIN_ROWS_FOR_TRAIN:
        raise ValueError(f"Too few training rows: {len(df)}.")

    X  = df[feature_cols].astype(float)
    y1 = df["impact_label"].astype(int)
    y2 = df["magnitude_label"].astype(float)

    unique_labels = sorted(y1.unique())
    label_to_idx  = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label  = {idx: lbl for lbl, idx in label_to_idx.items()}
    y1_enc        = y1.map(label_to_idx)
    clf_params    = {**LGB_CLF_PARAMS, "num_class": len(unique_labels)}

    # Stage 1 – initial
    log.info(f"Stage 1: classifier ({len(X)} rows, {len(feature_cols)} features)...")
    clf_init = lgb.LGBMClassifier(**clf_params)
    log.info(f"  CV macro-F1: {_cv_score(clf_params, True, X, y1_enc):.4f}")
    clf_init.fit(X, y1_enc)

    # SHAP selection + retrain
    selected  = _shap_select(clf_init, X)
    X_sel     = X[selected]
    clf_final = lgb.LGBMClassifier(**clf_params)
    clf_final.fit(X_sel, y1_enc)
    log.info(f"  Stage 1 retrained CV macro-F1: {_cv_score(clf_params, True, X_sel, y1_enc):.4f}")

    # Stage 2
    log.info("Stage 2: magnitude regressor...")
    reg = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    reg.fit(X_sel, y2)
    log.info(f"  Stage 2 CV MAE: {-_cv_score(LGB_REG_PARAMS, False, X_sel, y2):.6f}")

    for path, obj in [
        (STAGE1_MODEL_PATH, clf_final),
        (STAGE2_MODEL_PATH, reg),
        (FEATURES_PATH,     selected),
        (CLASS_MAP_PATH,    idx_to_label),
    ]:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    log.info(f"Models saved to {MODEL_DIR}/")
    return clf_final, reg, selected


# ─────────────────────────────────────────────
# 8. PREDICTION
# ─────────────────────────────────────────────

def load_models() -> tuple:
    for p in [STAGE1_MODEL_PATH, STAGE2_MODEL_PATH, FEATURES_PATH, CLASS_MAP_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing – run --train first.")
    with open(STAGE1_MODEL_PATH, "rb") as f: clf          = pickle.load(f)
    with open(STAGE2_MODEL_PATH, "rb") as f: reg          = pickle.load(f)
    with open(FEATURES_PATH,     "rb") as f: selected     = pickle.load(f)
    with open(CLASS_MAP_PATH,    "rb") as f: idx_to_label = pickle.load(f)
    return clf, reg, selected, idx_to_label


def predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clf, reg, selected, idx_to_label = load_models()
    missing = [c for c in selected if c not in df.columns]
    if missing:
        log.warning(f"{len(missing)} features absent – zero-filling: {missing[:6]}")
        for c in missing:
            df[c] = 0.0
    X          = df[selected].fillna(0).astype(float)
    proba      = clf.predict_proba(X)
    direction  = np.array([idx_to_label.get(int(i), 0) for i in proba.argmax(axis=1)])
    magnitude  = np.clip(reg.predict(X), 0, None)
    df         = df.copy()
    df["predicted_direction"] = direction.astype(int)
    df["predicted_magnitude"] = np.round(magnitude, 6)
    df["confidence_score"]    = np.round(proba.max(axis=1), 6)
    return df


# ─────────────────────────────────────────────
# 9. BACKTEST
# ─────────────────────────────────────────────

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Backtest: 0.1% one-way transaction cost...")
    parts: list[pd.DataFrame] = []
    asset_rets: dict[str, pd.Series] = {}

    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("timestamp").copy()
        if "forward_return_t1" not in grp.columns or grp["forward_return_t1"].isna().all():
            grp["strategy_return"] = np.nan
            parts.append(grp)
            continue
        fwd                    = grp["forward_return_t1"].fillna(0)
        pos                    = grp["predicted_direction"] * grp["predicted_magnitude"]
        grp["position"]        = pos.round(6)
        grp["strategy_return"] = (pos * fwd - pos.abs() * TXN_COST).round(6)
        sr                     = grp["strategy_return"]
        log.info(f"  [{asset:<12}] total={sr.sum():+.4f}  sharpe={(sr.mean()/(sr.std()+1e-9))*np.sqrt(252):+.2f}  win%={(sr>0).mean():.1%}")
        asset_rets[asset] = grp.set_index("timestamp")["strategy_return"]
        parts.append(grp)

    if asset_rets:
        port = pd.DataFrame(asset_rets).mean(axis=1)
        log.info(f"  [{'PORTFOLIO':<12}] total={port.sum():+.4f}  sharpe={(port.mean()/(port.std()+1e-9))*np.sqrt(252):+.2f}")

    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────
# 10. OUTPUT SCHEMA
# ─────────────────────────────────────────────

SCHEMA_FIELDS = ["timestamp", "asset", "predicted_direction", "predicted_magnitude",
                 "confidence_score", "regime", "strategy_return"]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"regime_label": "regime"}).copy()
    for f in SCHEMA_FIELDS:
        if f not in out.columns:
            out[f] = np.nan
    return out[SCHEMA_FIELDS].drop_duplicates(subset=["timestamp", "asset"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# 11. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run(mode: str, asset: Optional[str] = None) -> None:
    log.info("=" * 65)
    log.info(f"EventOracle – Modeling Pipeline  |  mode={mode.upper()}  |  asset={asset or 'ALL'}")
    log.info("=" * 65)

    try:
        client = get_mongo_client()
        log.info("Connected to MongoDB Atlas.")
    except Exception as e:
        log.error(f"MongoDB connection failed: {e}")
        return

    feat_df   = load_collection(client, FEATURES_COL, asset=asset)
    if feat_df.empty:
        log.error("No data in features collection. Exiting.")
        return
    social_df = load_collection(client, SOCIAL_COL, asset=asset)
    df        = merge_sources(feat_df, social_df)

    # Cross-asset: always pivot from full dataset even in single-asset mode
    log.info("Adding cross-asset features...")
    if asset:
        full_df     = merge_sources(load_collection(client, FEATURES_COL), load_collection(client, SOCIAL_COL))
        full_df     = add_cross_asset_features(full_df)
        new_cols    = [c for c in full_df.columns if "_xret" in c]
        cross_slice = full_df[full_df["asset"] == asset][["timestamp"] + new_cols]
        df          = df.merge(cross_slice, on="timestamp", how="left")
    else:
        df = add_cross_asset_features(df)

    log.info("Detecting market regimes...")
    df = add_regime_labels(df)

    if mode == "train":
        log.info("Labeling events...")
        df = label_events(df)

    log.info("Building feature matrix...")
    fit_enc = (not LABEL_ENCODER_PATH.exists()) or (mode == "train")
    enc: Optional[LabelEncoder] = None
    if not fit_enc:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            enc = pickle.load(f)

    df, feature_cols, enc = build_feature_matrix(df, encoder=enc, fit_encoder=fit_enc)

    if mode == "train":
        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(enc, f)
        train(df, feature_cols)
        log.info("Generating in-sample predictions for backtest...")
        df = predict(df, feature_cols)
        df = backtest(df)
    else:
        log.info("Generating predictions...")
        df = predict(df, feature_cols)
        has_fwd = "forward_return_t1" in df.columns and df["forward_return_t1"].notna().any()
        df = backtest(df) if has_fwd else df.assign(strategy_return=np.nan)

    output_df = build_output(df)
    log.info(f"Output: {len(output_df)} rows across {output_df['asset'].nunique()} asset(s).")
    upsert_predictions(client, output_df)
    log.info("=" * 65)
    log.info("Pipeline Complete.")
    log.info("=" * 65)


# ─────────────────────────────────────────────
# 12. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EventOracle Modeling Pipeline (P9) – Event Labeling + LightGBM + SHAP + Backtest"
    )
    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument("--train",   action="store_true", help="Train models + SHAP selection + backtest.")
    grp.add_argument("--predict", action="store_true", help="Load models + generate predictions.")
    parser.add_argument("--asset", type=str, default=None, choices=KNOWN_ASSETS,
                        metavar="ASSET", help=f"Single-asset mode. Choices: {KNOWN_ASSETS}")
    args = parser.parse_args()
    run(mode="train" if args.train else "predict", asset=args.asset)