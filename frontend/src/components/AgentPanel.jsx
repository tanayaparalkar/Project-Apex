import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { runMarketAgent } from "../api";

/**
 * AgentPanel – Runs and displays output from market_agent.py
 * Data source: POST /api/agent/<asset>
 *   → api_server.py calls market_agent.format_system_prompt()
 *   → Sends prompt to Gemini 2.5 Flash
 *   → Returns Markdown report
 *
 * The prompt is built from:
 *   - features collection (rsi, volatility, returns)
 *   - social_anomaly collection (sentiment, buzz, anomaly flag)
 *   - predictions collection (direction, magnitude, confidence, regime)
 */
export default function AgentPanel({ asset }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const data = await runMarketAgent(asset);
      setReport(data.report || "No report generated.");
    } catch (err) {
      setError(err.message || "Failed to generate report.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="agent-section">
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: "1rem" }}>
        <button className="agent-btn" onClick={handleRun} disabled={loading || !asset}>
          {loading ? (
            <>
              <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
              Analyzing {asset}...
            </>
          ) : (
            <>
              ✨ Run AI Market Agent
            </>
          )}
        </button>
        {asset && (
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Powered by Gemini 2.5 Flash · Analyzes features + social + prediction data
          </span>
        )}
      </div>

      {error && <div className="error-msg">⚠️ {error}</div>}

      {report && (
        <div className="agent-report">
          <ReactMarkdown>{report}</ReactMarkdown>
        </div>
      )}

      {!report && !loading && !error && (
        <div style={{
          padding: "2rem",
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: "0.85rem",
          background: "var(--bg-secondary)",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--border-subtle)",
        }}>
          <div style={{ fontSize: "2rem", marginBottom: "0.75rem", opacity: 0.4 }}>🧠</div>
          Click <strong>"Run AI Market Agent"</strong> to generate a comprehensive trading analysis
          for <strong>{asset}</strong> using data from all pipelines.
        </div>
      )}
    </div>
  );
}
