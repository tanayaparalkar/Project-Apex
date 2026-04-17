/**
 * api.js – EventOracle API Client
 * All fetch calls to the Flask backend (api_server.py).
 * Maps 1:1 with the MongoDB collections served by the backend.
 */

const API_BASE = "http://localhost:5000/api";

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

/** GET /api/assets → list of all tracked assets with latest close/returns */
export const fetchAssets = () => fetchJSON(`${API_BASE}/assets`);

/** GET /api/prices/<asset>?limit=N → OHLCV + technicals from prices collection */
export const fetchPrices = (asset, limit = 500) =>
  fetchJSON(`${API_BASE}/prices/${asset}?limit=${limit}`);

/** GET /api/features/<asset>?limit=N → merged feature rows from features collection */
export const fetchFeatures = (asset, limit = 30) =>
  fetchJSON(`${API_BASE}/features/${asset}?limit=${limit}`);

/** GET /api/news/<asset>?limit=N → filtered news from news collection */
export const fetchNewsByAsset = (asset, limit = 30) =>
  fetchJSON(`${API_BASE}/news/${asset}?limit=${limit}`);

/** GET /api/news?limit=N → all recent news */
export const fetchAllNews = (limit = 50) =>
  fetchJSON(`${API_BASE}/news?limit=${limit}`);

/** GET /api/predictions/<asset> → model output from predictions collection */
export const fetchPredictions = (asset, limit = 30) =>
  fetchJSON(`${API_BASE}/predictions/${asset}?limit=${limit}`);

/** GET /api/social/<asset> → social anomaly data from social_anomaly collection */
export const fetchSocial = (asset, limit = 30) =>
  fetchJSON(`${API_BASE}/social/${asset}?limit=${limit}`);

/** GET /api/summary/<asset> → combined summary (price + prediction + social) */
export const fetchSummary = (asset) =>
  fetchJSON(`${API_BASE}/summary/${asset}`);

/** POST /api/agent/<asset> → run market_agent.py via Gemini and return report */
export const runMarketAgent = async (asset) => {
  const res = await fetch(`${API_BASE}/agent/${asset}`, { method: "POST" });
  if (!res.ok) throw new Error(`Agent API ${res.status}`);
  return res.json();
};
