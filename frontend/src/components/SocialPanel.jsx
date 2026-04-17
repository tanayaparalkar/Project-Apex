import React from "react";

/**
 * SocialPanel – social_anomaly.py output display
 * Data source: /api/social/<asset> → social_anomaly.py → MongoDB social_anomaly collection
 *
 * Shows:
 *   - social_sentiment_score (weighted: 50% sentiment + 30% velocity_z + 20% volume_z)
 *   - social_buzz_intensity (z-score of news volume)
 *   - sentiment_momentum (sentiment change over 3 days)
 *   - sentiment_volatility (5-day rolling std)
 *   - anomaly_score & anomaly_flag (Isolation Forest output)
 *   - market_stress_score & composite_anomaly_flag (cross-domain flags)
 */
export default function SocialPanel({ social, loading }) {
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading social signals...</span>
      </div>
    );
  }

  if (!social || social.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🔍</div>
        <p>No social anomaly data. Run <code>python social_anomaly.py</code> first.</p>
      </div>
    );
  }

  const latest = social[social.length - 1];

  const items = [
    {
      label: "Social Sentiment",
      value: latest.social_sentiment_score,
      format: (v) => v != null ? Number(v).toFixed(4) : "—",
      color: (v) => v > 0.05 ? "var(--accent-emerald)" : v < -0.05 ? "var(--accent-rose)" : "var(--text-primary)",
    },
    {
      label: "Buzz Intensity",
      value: latest.social_buzz_intensity,
      format: (v) => v != null ? Number(v).toFixed(3) : "—",
      color: () => "var(--accent-cyan)",
    },
    {
      label: "Sentiment Momentum",
      value: latest.sentiment_momentum,
      format: (v) => v != null ? (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(4) : "—",
      color: (v) => v > 0 ? "var(--accent-emerald)" : v < 0 ? "var(--accent-rose)" : "var(--text-primary)",
    },
    {
      label: "Sentiment Volatility",
      value: latest.sentiment_volatility,
      format: (v) => v != null ? Number(v).toFixed(4) : "—",
      color: () => "var(--accent-amber)",
    },
    {
      label: "Anomaly Score (IF)",
      value: latest.anomaly_score,
      format: (v) => v != null ? Number(v).toFixed(4) : "—",
      color: (v) => v != null && v < 0 ? "var(--accent-rose)" : "var(--accent-emerald)",
    },
    {
      label: "Market Stress",
      value: latest.market_stress_score,
      format: (v) => v != null ? String(v) + " / 3" : "—",
      color: (v) => v >= 2 ? "var(--accent-rose)" : v == 1 ? "var(--accent-amber)" : "var(--accent-emerald)",
    },
  ];

  const anomalyActive = latest.composite_anomaly_flag === 1;
  const isoAnomaly = latest.anomaly_flag === 1;

  return (
    <div>
      {/* Alert Badges */}
      <div style={{ display: "flex", gap: 10, marginBottom: "1rem", flexWrap: "wrap" }}>
        <div
          className={`direction-badge ${anomalyActive ? "down" : "up"}`}
          style={{ fontSize: "0.78rem" }}
        >
          {anomalyActive ? "⚠️ ANOMALY DETECTED" : "✅ MARKET NORMAL"}
        </div>
        {isoAnomaly && (
          <div className="direction-badge down" style={{ fontSize: "0.78rem" }}>
            🔴 Isolation Forest Flag
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "0.75rem",
      }}>
        {items.map((item, idx) => (
          <div key={idx} className="pred-item">
            <div className="pred-label">{item.label}</div>
            <div
              className="pred-value"
              style={{
                color: item.color(item.value),
                fontSize: "1.15rem",
              }}
            >
              {item.format(item.value)}
            </div>
          </div>
        ))}
      </div>

      {/* Timestamp */}
      <div style={{
        marginTop: "1rem",
        fontSize: "0.72rem",
        color: "var(--text-muted)",
        textAlign: "right",
        fontFamily: "var(--font-mono)",
      }}>
        Last updated: {latest.timestamp
          ? new Date(latest.timestamp).toLocaleString("en-IN", {
              day: "numeric", month: "short", year: "numeric",
              hour: "2-digit", minute: "2-digit",
            })
          : "—"
        }
      </div>
    </div>
  );
}
