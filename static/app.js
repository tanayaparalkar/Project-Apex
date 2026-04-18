/**
 * app.js – EventOracle Dashboard (Vanilla JS)
 *
 * API Base URL toggles between:
 *   - VPC backend:  http://34.180.15.52:5000/api
 *   - Localhost:    http://localhost:5000/api
 *   - Same-origin:  /api  (when served by Flask on Render)
 */

// ─── API Configuration ─────────────────────────
const API_CONFIGS = {
  vpc:       "http://34.180.15.52:5000/api",
  local:     "http://localhost:5000/api",
  sameOrigin: "/api",
};

// Auto-detect: if page is served from localhost → use local, if from Render → same-origin
function detectAPIBase() {
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") return API_CONFIGS.local;
  return API_CONFIGS.sameOrigin;  // Render / production
}

let API_BASE = detectAPIBase();

// Override: to force VPC, uncomment:
// API_BASE = API_CONFIGS.vpc;

// ─── Assets ─────────────────────────────────────
const ASSETS = [
  { name: "NIFTY",     icon: "📈" },
  { name: "BANKNIFTY", icon: "🏦" },
  { name: "NIFTYIT",   icon: "💻" },
  { name: "NIFTYMETAL",icon: "⚙️" },
  { name: "INRUSD",    icon: "💱" },
  { name: "BRENT",     icon: "🛢️" },
  { name: "GOLD",      icon: "🥇" },
  { name: "SILVER",    icon: "🥈" },
  { name: "PLATINUM",  icon: "💎" },
  { name: "BITCOIN",   icon: "₿"  },
];

let selectedAsset = "NIFTY";
let chartInstances = {};

// ─── Init ───────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("header-date").textContent = new Date().toLocaleDateString("en-IN", {
    weekday: "short", day: "numeric", month: "short", year: "numeric"
  });
  renderAssetBar();
  loadAll(selectedAsset);
});

// ─── Asset Bar ──────────────────────────────────
function renderAssetBar() {
  const bar = document.getElementById("asset-bar");
  bar.innerHTML = ASSETS.map(a =>
    `<button class="asset-btn ${a.name === selectedAsset ? 'active' : ''}" data-asset="${a.name}" onclick="selectAsset('${a.name}')">
      <span style="margin-right:6px">${a.icon}</span>${a.name}
    </button>`
  ).join("");
}

function selectAsset(asset) {
  selectedAsset = asset;
  renderAssetBar();
  loadAll(asset);
}

// ─── API Helpers ────────────────────────────────
async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

// ─── Load All Data ──────────────────────────────
function loadAll(asset) {
  document.getElementById("price-chart-title").textContent = `${asset} — Price History`;
  document.getElementById("news-title").textContent = `Latest News — ${asset}`;
  document.getElementById("agent-title").textContent = `AI Market Agent — ${asset}`;
  document.getElementById("agent-output").innerHTML = `<div class="agent-placeholder"><div class="empty-icon">🧠</div>Click <strong>"Run AI Market Agent"</strong> to generate a comprehensive trading analysis for <strong>${asset}</strong>.</div>`;

  loadSummary(asset);
  loadPrices(asset);
  loadPredictions(asset);
  loadNews(asset);
  loadSocial(asset);
  loadFeatures(asset);
}

// ─── Summary Stats ──────────────────────────────
async function loadSummary(asset) {
  try {
    const s = await fetchJSON(`/summary/${asset}`);
    const price = s.price?.close;
    const returns = s.price?.returns;
    const dir = s.prediction?.direction;
    const conf = s.prediction?.confidence;
    const anomaly = s.social?.anomaly_flag;
    const stress = s.social?.market_stress;

    // Price
    document.getElementById("stat-price-val").textContent = price != null ? formatPrice(price) : "—";
    document.getElementById("stat-price-sub").textContent = s.price?.timestamp ? new Date(s.price.timestamp).toLocaleDateString("en-IN", { month: "short", day: "numeric" }) : "—";

    // Returns
    const retEl = document.getElementById("stat-returns-val");
    const retCard = document.getElementById("stat-returns");
    if (returns != null) {
      retEl.textContent = `${returns > 0 ? "+" : ""}${(returns * 100).toFixed(3)}%`;
      retEl.className = `stat-value ${returns > 0 ? "positive" : returns < 0 ? "negative" : "neutral"}`;
      retCard.className = `stat-card ${returns > 0 ? "green" : "rose"}`;
    } else { retEl.textContent = "—"; }
    document.getElementById("stat-returns-sub").textContent = returns != null ? `${returns > 0 ? "▲" : "▼"} Day change` : "—";

    // Direction
    const dirEl = document.getElementById("stat-dir-val");
    if (dir === 1)       { dirEl.textContent = "▲ BUY";  dirEl.className = "stat-value positive"; }
    else if (dir === -1) { dirEl.textContent = "▼ SELL"; dirEl.className = "stat-value negative"; }
    else if (dir === 0)  { dirEl.textContent = "◆ HOLD"; dirEl.className = "stat-value neutral"; }
    else                 { dirEl.textContent = "—"; }
    document.getElementById("stat-dir-sub").textContent = conf != null ? `${(conf * 100).toFixed(1)}% conf` : "—";

    // Anomaly
    const anomEl = document.getElementById("stat-anomaly-val");
    const anomCard = document.getElementById("stat-anomaly");
    if (anomaly) {
      anomEl.textContent = "⚠ ALERT"; anomEl.className = "stat-value negative";
      anomCard.className = "stat-card rose";
    } else {
      anomEl.textContent = "✓ NORMAL"; anomEl.className = "stat-value positive";
      anomCard.className = "stat-card green";
    }
    document.getElementById("stat-anomaly-sub").textContent = `Stress: ${stress ?? "—"}/3`;
  } catch (e) { console.error("Summary error:", e); }
}

// ─── Price Chart ────────────────────────────────
async function loadPrices(asset) {
  try {
    const data = await fetchJSON(`/prices/${asset}?limit=500`);
    if (!data.length) return;

    const labels = data.map(r => fmtDate(r.timestamp));
    const closes = data.map(r => r.close);
    const highs  = data.map(r => r.high);
    const lows   = data.map(r => r.low);

    destroyChart("priceChart");
    const ctx = document.getElementById("priceChart").getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 0, 380);
    grad.addColorStop(0, "rgba(59,130,246,0.25)");
    grad.addColorStop(1, "rgba(59,130,246,0.02)");

    chartInstances["priceChart"] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Close", data: closes, borderColor: "#3b82f6", borderWidth: 2.5, fill: true, backgroundColor: grad, pointRadius: 0, pointHoverRadius: 5, tension: 0.3 },
          { label: "High", data: highs, borderColor: "rgba(16,185,129,0.4)", borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false, tension: 0.3 },
          { label: "Low",  data: lows,  borderColor: "rgba(244,63,94,0.4)",  borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false, tension: 0.3 },
        ],
      },
      options: chartOpts("Price"),
    });
  } catch (e) { console.error("Prices error:", e); }
}

// ─── Predictions ────────────────────────────────
async function loadPredictions(asset) {
  const body = document.getElementById("predictions-body");
  try {
    const data = await fetchJSON(`/predictions/${asset}?limit=30`);
    if (!data.length) { body.innerHTML = `<div class="empty-state"><div class="empty-icon">🤖</div><p>No model predictions available.</p></div>`; return; }

    const latest = data[data.length - 1];
    const dir = latest.predicted_direction;
    const dirText = dir === 1 ? "▲ BULLISH" : dir === -1 ? "▼ BEARISH" : "◆ NEUTRAL";
    const dirCls  = dir === 1 ? "up" : dir === -1 ? "down" : "flat";
    const conf = latest.confidence_score != null ? (latest.confidence_score * 100).toFixed(1) : "—";
    const mag  = latest.predicted_magnitude != null ? (latest.predicted_magnitude * 100).toFixed(3) : "—";
    const regime = latest.regime === 0 ? "Normal" : latest.regime === 1 ? "Volatile" : latest.regime === 2 ? "Mean-Rev" : "—";
    const strat = latest.strategy_return != null ? (latest.strategy_return * 100).toFixed(3) : "—";
    const stratCls = latest.strategy_return > 0 ? "up" : latest.strategy_return < 0 ? "down" : "flat";

    body.innerHTML = `
      <div class="pred-grid">
        <div class="pred-item"><div class="pred-label">Direction</div><div class="direction-badge ${dirCls}">${dirText}</div></div>
        <div class="pred-item"><div class="pred-label">Magnitude</div><div class="pred-value ${dirCls}">${mag}%</div></div>
        <div class="pred-item"><div class="pred-label">Confidence</div><div class="pred-value flat">${conf}%</div></div>
      </div>
      <div class="pred-grid">
        <div class="pred-item"><div class="pred-label">Regime</div><div class="pred-value flat">${regime}</div></div>
        <div class="pred-item"><div class="pred-label">Strategy Return</div><div class="pred-value ${stratCls}">${strat}%</div></div>
        <div class="pred-item"><div class="pred-label">Model</div><div class="pred-value flat" style="font-size:0.85rem">LightGBM</div></div>
      </div>
      <div class="chart-wrap-sm" style="margin-top:1rem"><canvas id="confChart"></canvas></div>`;

    // Confidence chart
    const confData = data.map(r => r.confidence_score != null ? +(r.confidence_score * 100).toFixed(1) : null);
    const confLabels = data.map(r => fmtDate(r.timestamp));
    const cCtx = document.getElementById("confChart").getContext("2d");
    destroyChart("confChart");
    chartInstances["confChart"] = new Chart(cCtx, {
      type: "line",
      data: { labels: confLabels, datasets: [{ label: "Confidence %", data: confData, borderColor: "#8b5cf6", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false }] },
      options: chartOpts("Confidence"),
    });
  } catch (e) { body.innerHTML = `<div class="error-msg">⚠️ ${e.message}</div>`; }
}

// ─── News ───────────────────────────────────────
async function loadNews(asset) {
  const list = document.getElementById("news-list");
  try {
    const data = await fetchJSON(`/news/${asset}?limit=30`);
    if (!data.length) { list.innerHTML = `<li class="empty-state"><div class="empty-icon">📰</div><p>No news available</p></li>`; return; }

    list.innerHTML = data.map(item => {
      const sentCls = (item.sentiment_label || "neutral").toLowerCase();
      const sentBadge = item.sentiment_label ? `<span class="news-sentiment ${sentCls}">${item.sentiment_label}${item.sentiment_score != null ? ` (${Number(item.sentiment_score).toFixed(2)})` : ""}</span>` : "";
      const tagBadge = item.event_type && item.event_type !== "Other" ? `<span class="news-tag">${item.event_type}</span>` : "";
      const headline = item.link ? `<a href="${item.link}" target="_blank" rel="noopener">${item.headline}</a>` : item.headline;
      return `<li class="news-item">
        <div class="news-headline">${headline}</div>
        <div class="news-meta"><span>${item.source || ""}</span><span>•</span><span>${fmtTime(item.timestamp)}</span>${sentBadge}${tagBadge}</div>
      </li>`;
    }).join("");
  } catch (e) { list.innerHTML = `<li class="error-msg">⚠️ ${e.message}</li>`; }
}

// ─── Social Anomaly ─────────────────────────────
async function loadSocial(asset) {
  const body = document.getElementById("social-body");
  try {
    const data = await fetchJSON(`/social/${asset}?limit=30`);
    if (!data.length) { body.innerHTML = `<div class="empty-state"><div class="empty-icon">🔍</div><p>No social anomaly data.</p></div>`; return; }

    const l = data[data.length - 1];
    const anomalyActive = l.composite_anomaly_flag === 1;
    const isoAnomaly = l.anomaly_flag === 1;

    const items = [
      { label: "Social Sentiment", val: l.social_sentiment_score, fmt: v => v != null ? Number(v).toFixed(4) : "—", clr: v => v > 0.05 ? "up" : v < -0.05 ? "down" : "flat" },
      { label: "Buzz Intensity", val: l.social_buzz_intensity, fmt: v => v != null ? Number(v).toFixed(3) : "—", clr: () => "flat" },
      { label: "Sentiment Momentum", val: l.sentiment_momentum, fmt: v => v != null ? (v >= 0 ? "+" : "") + Number(v).toFixed(4) : "—", clr: v => v > 0 ? "up" : v < 0 ? "down" : "flat" },
      { label: "Sentiment Volatility", val: l.sentiment_volatility, fmt: v => v != null ? Number(v).toFixed(4) : "—", clr: () => "flat" },
      { label: "Anomaly Score (IF)", val: l.anomaly_score, fmt: v => v != null ? Number(v).toFixed(4) : "—", clr: v => v != null && v < 0 ? "down" : "up" },
      { label: "Market Stress", val: l.market_stress_score, fmt: v => v != null ? v + " / 3" : "—", clr: v => v >= 2 ? "down" : v == 1 ? "flat" : "up" },
    ];

    body.innerHTML = `
      <div class="social-badges">
        <div class="direction-badge ${anomalyActive ? 'down' : 'up'}">${anomalyActive ? '⚠️ ANOMALY DETECTED' : '✅ MARKET NORMAL'}</div>
        ${isoAnomaly ? '<div class="direction-badge down">🔴 Isolation Forest Flag</div>' : ''}
      </div>
      <div class="social-grid">
        ${items.map(it => `<div class="pred-item"><div class="pred-label">${it.label}</div><div class="pred-value ${it.clr(it.val)}" style="font-size:1.15rem">${it.fmt(it.val)}</div></div>`).join("")}
      </div>
      <div class="social-ts">Last updated: ${l.timestamp ? new Date(l.timestamp).toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</div>`;
  } catch (e) { body.innerHTML = `<div class="error-msg">⚠️ ${e.message}</div>`; }
}

// ─── Technical Indicators ───────────────────────
async function loadFeatures(asset) {
  try {
    const data = await fetchJSON(`/features/${asset}?limit=60`);
    if (!data.length) return;

    const labels = data.map(r => fmtDate(r.timestamp));

    // RSI
    destroyChart("rsiChart");
    chartInstances["rsiChart"] = new Chart(document.getElementById("rsiChart"), {
      type: "line",
      data: { labels, datasets: [{ label: "RSI", data: data.map(r => r.rsi), borderColor: "#f59e0b", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false }] },
      options: { ...chartOpts("RSI"), scales: { ...chartOpts("RSI").scales, y: { ...chartOpts("RSI").scales.y, min: 0, max: 100 } } },
      plugins: [{ id: "rsiLines", afterDraw(chart) {
        const {ctx, chartArea, scales} = chart;
        [30, 70].forEach(v => {
          const y = scales.y.getPixelForValue(v);
          ctx.save(); ctx.strokeStyle = v === 70 ? "rgba(244,63,94,0.3)" : "rgba(16,185,129,0.3)";
          ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(chartArea.left, y); ctx.lineTo(chartArea.right, y); ctx.stroke(); ctx.restore();
        });
      }}],
    });

    // Volatility
    destroyChart("volChart");
    const vGrad = document.getElementById("volChart").getContext("2d").createLinearGradient(0,0,0,160);
    vGrad.addColorStop(0,"rgba(244,63,94,0.3)"); vGrad.addColorStop(1,"rgba(244,63,94,0.02)");
    chartInstances["volChart"] = new Chart(document.getElementById("volChart"), {
      type: "line",
      data: { labels, datasets: [{ label: "Volatility %", data: data.map(r => r.rolling_volatility != null ? +(r.rolling_volatility*100).toFixed(3) : null), borderColor: "#f43f5e", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: vGrad }] },
      options: chartOpts("Vol"),
    });

    // Sentiment
    destroyChart("sentChart");
    const sGrad = document.getElementById("sentChart").getContext("2d").createLinearGradient(0,0,0,160);
    sGrad.addColorStop(0,"rgba(16,185,129,0.3)"); sGrad.addColorStop(1,"rgba(16,185,129,0.02)");
    chartInstances["sentChart"] = new Chart(document.getElementById("sentChart"), {
      type: "line",
      data: { labels, datasets: [{ label: "Sentiment", data: data.map(r => r.avg_sentiment), borderColor: "#10b981", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: sGrad }] },
      options: chartOpts("Sent"),
    });

    // Volume Z-Score
    destroyChart("vzChart");
    chartInstances["vzChart"] = new Chart(document.getElementById("vzChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Vol Z-Score", data: data.map(r => r.volume_zscore != null ? +Number(r.volume_zscore).toFixed(2) : null),
        backgroundColor: data.map(r => {
          const v = r.volume_zscore || 0;
          return v > 2 ? "rgba(244,63,94,0.6)" : v < -2 ? "rgba(16,185,129,0.6)" : "rgba(99,102,241,0.6)";
        }),
        borderRadius: 2 }] },
      options: chartOpts("VZ"),
    });
  } catch (e) { console.error("Features error:", e); }
}

// ─── AI Agent ───────────────────────────────────
async function runAgent() {
  const btn = document.getElementById("agent-btn");
  const output = document.getElementById("agent-output");
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:18px;height:18px;border-width:2px"></div> Analyzing ${selectedAsset}...`;
  output.innerHTML = `<div class="loading-container"><div class="spinner"></div><span>Generating AI analysis for ${selectedAsset}…</span></div>`;

  try {
    const res = await fetch(`${API_BASE}/agent/${selectedAsset}`, { method: "POST" });
    const data = await res.json();
    if (data.report) {
      output.innerHTML = `<div class="agent-report">${marked.parse(data.report)}</div>`;
    } else {
      output.innerHTML = `<div class="error-msg">⚠️ ${data.error || "No report generated."}</div>`;
    }
  } catch (e) {
    output.innerHTML = `<div class="error-msg">⚠️ ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "✨ Run AI Market Agent";
  }
}

// ─── Chart Helpers ──────────────────────────────
function chartOpts(label) {
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: { intersect: false, mode: "index" },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#1e293b", borderColor: "rgba(148,163,184,0.15)", borderWidth: 1,
        titleFont: { family: "'Inter'", size: 11, weight: 600 },
        bodyFont: { family: "'JetBrains Mono'", size: 11 },
        padding: 10, cornerRadius: 10,
      },
    },
    scales: {
      x: { ticks: { color: "#64748b", font: { size: 9 }, maxRotation: 0, autoSkipPadding: 20 }, grid: { color: "rgba(148,163,184,0.06)", drawBorder: false } },
      y: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "rgba(148,163,184,0.06)", drawBorder: false } },
    },
  };
}

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

// ─── Formatting ─────────────────────────────────
function fmtDate(ts) {
  if (!ts) return "";
  return new Date(ts).toLocaleDateString("en-IN", { month: "short", day: "numeric" });
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  const diff = Date.now() - d.getTime();
  const hrs = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (hrs < 1) return "Just now";
  if (hrs < 24) return `${hrs}h ago`;
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString("en-IN", { month: "short", day: "numeric", year: "numeric" });
}

function formatPrice(val) {
  if (val == null) return "—";
  if (val >= 100000) return `₹${(val/1000).toFixed(1)}K`;
  if (val >= 1000) return `₹${val.toFixed(2)}`;
  return val.toFixed(4);
}
