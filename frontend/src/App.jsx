import React, { useState, useEffect, useCallback } from "react";
import PriceChart from "./components/PriceChart";
import NewsFeed from "./components/NewsFeed";
import PredictionPanel from "./components/PredictionPanel";
import AgentPanel from "./components/AgentPanel";
import TechnicalsPanel from "./components/TechnicalsPanel";
import SocialPanel from "./components/SocialPanel";
import {
  fetchAssets,
  fetchPrices,
  fetchFeatures,
  fetchNewsByAsset,
  fetchPredictions,
  fetchSocial,
  fetchSummary,
} from "./api";

const ASSETS = [
  "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
  "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
];

const ASSET_ICONS = {
  NIFTY: "📈", BANKNIFTY: "🏦", NIFTYIT: "💻", NIFTYMETAL: "⚙️",
  INRUSD: "💱", BRENT: "🛢️", GOLD: "🥇", SILVER: "🥈",
  PLATINUM: "💎", BITCOIN: "₿",
};

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState("NIFTY");
  const [prices, setPrices] = useState([]);
  const [features, setFeatures] = useState([]);
  const [news, setNews] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [social, setSocial] = useState([]);
  const [summary, setSummary] = useState(null);

  const [loading, setLoading] = useState({
    prices: false,
    features: false,
    news: false,
    predictions: false,
    social: false,
    summary: false,
  });

  const [errors, setErrors] = useState({});

  const loadData = useCallback(async (asset) => {
    setLoading({
      prices: true, features: true, news: true,
      predictions: true, social: true, summary: true,
    });
    setErrors({});

    // Prices
    fetchPrices(asset, 500)
      .then((data) => { setPrices(data); setLoading((p) => ({ ...p, prices: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, prices: e.message })); setLoading((p) => ({ ...p, prices: false })); });

    // Features
    fetchFeatures(asset, 60)
      .then((data) => { setFeatures(data); setLoading((p) => ({ ...p, features: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, features: e.message })); setLoading((p) => ({ ...p, features: false })); });

    // News
    fetchNewsByAsset(asset, 30)
      .then((data) => { setNews(data); setLoading((p) => ({ ...p, news: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, news: e.message })); setLoading((p) => ({ ...p, news: false })); });

    // Predictions
    fetchPredictions(asset, 30)
      .then((data) => { setPredictions(data); setLoading((p) => ({ ...p, predictions: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, predictions: e.message })); setLoading((p) => ({ ...p, predictions: false })); });

    // Social
    fetchSocial(asset, 30)
      .then((data) => { setSocial(data); setLoading((p) => ({ ...p, social: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, social: e.message })); setLoading((p) => ({ ...p, social: false })); });

    // Summary
    fetchSummary(asset)
      .then((data) => { setSummary(data); setLoading((p) => ({ ...p, summary: false })); })
      .catch((e) => { setErrors((p) => ({ ...p, summary: e.message })); setLoading((p) => ({ ...p, summary: false })); });
  }, []);

  useEffect(() => {
    loadData(selectedAsset);
  }, [selectedAsset, loadData]);

  const handleAssetSelect = (asset) => {
    setSelectedAsset(asset);
    // Reset states
    setPrices([]);
    setFeatures([]);
    setNews([]);
    setPredictions([]);
    setSocial([]);
    setSummary(null);
  };

  // Summary stat helpers
  const latestPrice = summary?.price?.close;
  const latestReturns = summary?.price?.returns;
  const direction = summary?.prediction?.direction;
  const confidence = summary?.prediction?.confidence;

  const formatPrice = (val) => {
    if (val == null) return "—";
    if (val >= 100000) return `₹${(val / 1000).toFixed(1)}K`;
    if (val >= 1000) return `₹${val.toFixed(2)}`;
    return val.toFixed(4);
  };

  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">⚡</div>
            <div className="logo-text">
              <span>EventOracle</span>
            </div>
          </div>
          <div className="header-status">
            <div className="status-dot" />
            <span>Live Dashboard</span>
            <span style={{ margin: "0 6px", opacity: 0.3 }}>|</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
              {new Date().toLocaleDateString("en-IN", {
                weekday: "short", day: "numeric", month: "short", year: "numeric"
              })}
            </span>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main-content">
        {/* Asset Selector */}
        <div className="asset-bar">
          {ASSETS.map((asset) => (
            <button
              key={asset}
              className={`asset-btn ${selectedAsset === asset ? "active" : ""}`}
              onClick={() => handleAssetSelect(asset)}
            >
              <span style={{ marginRight: 6 }}>{ASSET_ICONS[asset]}</span>
              {asset}
            </button>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="stat-grid">
          <div className="stat-card blue">
            <div className="stat-label">Close Price</div>
            <div className="stat-value neutral">{formatPrice(latestPrice)}</div>
            <div className="stat-sub">
              {summary?.price?.timestamp
                ? new Date(summary.price.timestamp).toLocaleDateString("en-IN", { month: "short", day: "numeric" })
                : "—"
              }
            </div>
          </div>

          <div className={`stat-card ${latestReturns > 0 ? "green" : latestReturns < 0 ? "rose" : "amber"}`}>
            <div className="stat-label">Daily Returns</div>
            <div className={`stat-value ${latestReturns > 0 ? "positive" : latestReturns < 0 ? "negative" : "neutral"}`}>
              {latestReturns != null
                ? `${latestReturns > 0 ? "+" : ""}${(latestReturns * 100).toFixed(3)}%`
                : "—"
              }
            </div>
            <div className="stat-sub">
              {latestReturns > 0 ? "▲" : latestReturns < 0 ? "▼" : "—"} Day change
            </div>
          </div>

          <div className="stat-card violet">
            <div className="stat-label">ML Direction</div>
            <div className={`stat-value ${direction === 1 ? "positive" : direction === -1 ? "negative" : "neutral"}`}>
              {direction === 1 ? "▲ BUY" : direction === -1 ? "▼ SELL" : direction === 0 ? "◆ HOLD" : "—"}
            </div>
            <div className="stat-sub">
              {confidence != null ? `${(confidence * 100).toFixed(1)}% conf` : "—"}
            </div>
          </div>

          <div className={`stat-card ${summary?.social?.anomaly_flag ? "rose" : "green"}`}>
            <div className="stat-label">Anomaly Status</div>
            <div className={`stat-value ${summary?.social?.anomaly_flag ? "negative" : "positive"}`}>
              {summary?.social?.anomaly_flag ? "⚠ ALERT" : "✓ NORMAL"}
            </div>
            <div className="stat-sub">
              Stress: {summary?.social?.market_stress ?? "—"}/3
            </div>
          </div>
        </div>

        {/* Price Chart */}
        <div className="card full-width" style={{ marginBottom: "1.25rem" }}>
          <div className="card-header">
            <div className="card-title">
              <div className="card-title-icon blue">📈</div>
              {selectedAsset} — Price History
            </div>
            <div className="card-badge">market_pipeline.py → prices</div>
          </div>
          <div className="card-body">
            {loading.prices ? (
              <div className="loading-container"><div className="spinner" /><span>Loading prices...</span></div>
            ) : errors.prices ? (
              <div className="error-msg">⚠️ {errors.prices}</div>
            ) : (
              <PriceChart data={prices} asset={selectedAsset} />
            )}
          </div>
        </div>

        {/* Two-column: Predictions + News */}
        <div className="dashboard-grid">
          {/* Predictions / modeling.py */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <div className="card-title-icon violet">🤖</div>
                Model Predictions
              </div>
              <div className="card-badge">modeling.py → predictions</div>
            </div>
            <div className="card-body">
              <PredictionPanel predictions={predictions} loading={loading.predictions} />
            </div>
          </div>

          {/* News Feed */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <div className="card-title-icon cyan">📰</div>
                Latest News — {selectedAsset}
              </div>
              <div className="card-badge">news_pipeline.py → news</div>
            </div>
            <div className="card-body">
              <NewsFeed news={news} loading={loading.news} />
            </div>
          </div>
        </div>

        {/* Two-column: Social Anomaly + Technicals */}
        <div className="dashboard-grid" style={{ marginTop: "1.25rem" }}>
          {/* Social Anomaly */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <div className="card-title-icon rose">🔍</div>
                Social & Anomaly Signals
              </div>
              <div className="card-badge">social_anomaly.py → social_anomaly</div>
            </div>
            <div className="card-body">
              <SocialPanel social={social} loading={loading.social} />
            </div>
          </div>

          {/* Technicals */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <div className="card-title-icon amber">📊</div>
                Technical Indicators
              </div>
              <div className="card-badge">cleaning_pipeline.py → features</div>
            </div>
            <div className="card-body">
              <TechnicalsPanel features={features} loading={loading.features} />
            </div>
          </div>
        </div>

        {/* AI Market Agent */}
        <div className="card full-width" style={{ marginTop: "1.25rem" }}>
          <div className="card-header">
            <div className="card-title">
              <div className="card-title-icon green">🧠</div>
              AI Market Agent — {selectedAsset}
            </div>
            <div className="card-badge">market_agent.py → Gemini 2.5 Flash</div>
          </div>
          <div className="card-body">
            <AgentPanel asset={selectedAsset} />
          </div>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer style={{
        textAlign: "center",
        padding: "2rem",
        fontSize: "0.75rem",
        color: "var(--text-muted)",
        borderTop: "1px solid var(--border-subtle)",
        marginTop: "2rem",
      }}>
        EventOracle — Market Intelligence Platform · Built with React + Flask + MongoDB
      </footer>
    </div>
  );
}
