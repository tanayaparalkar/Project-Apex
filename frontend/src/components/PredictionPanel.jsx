import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

/**
 * PredictionPanel – Displays output from modeling.py
 * Data source: /api/predictions/<asset> → modeling.py → MongoDB predictions collection
 *
 * Shows:
 *   - predicted_direction (from LightGBM Stage 1 classifier)
 *   - predicted_magnitude (from LightGBM Stage 2 regressor)
 *   - confidence_score (softmax probability from Stage 1)
 *   - regime (KMeans market regime: 0=Normal, 1=Volatile, 2=Mean-Reverting)
 *   - strategy_return (from backtest with 0.1% txn cost)
 */
export default function PredictionPanel({ predictions, loading }) {
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading predictions...</span>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🤖</div>
        <p>No model predictions available. Run <code>python modeling.py --predict</code> first.</p>
      </div>
    );
  }

  const latest = predictions[predictions.length - 1];

  const directionLabel = (d) => {
    if (d === 1) return { text: "▲ BULLISH", cls: "up" };
    if (d === -1) return { text: "▼ BEARISH", cls: "down" };
    return { text: "◆ NEUTRAL", cls: "flat" };
  };

  const regimeLabel = (r) => {
    if (r === 0) return "Normal";
    if (r === 1) return "Volatile";
    if (r === 2) return "Mean-Rev";
    return "—";
  };

  const dir = directionLabel(latest.predicted_direction);
  const conf = latest.confidence_score != null ? (latest.confidence_score * 100).toFixed(1) : "—";
  const mag = latest.predicted_magnitude != null ? (latest.predicted_magnitude * 100).toFixed(3) : "—";

  // Chart data for historical predictions
  const chartData = predictions.map((row) => ({
    date: row.timestamp
      ? new Date(row.timestamp).toLocaleDateString("en-IN", { month: "short", day: "numeric" })
      : "",
    confidence: row.confidence_score != null ? +(row.confidence_score * 100).toFixed(1) : null,
    stratReturn: row.strategy_return != null ? +(row.strategy_return * 100).toFixed(3) : null,
    direction: row.predicted_direction,
  }));

  return (
    <div>
      {/* Latest prediction summary */}
      <div className="prediction-grid" style={{ marginBottom: "1.25rem" }}>
        <div className="pred-item">
          <div className="pred-label">Direction</div>
          <div className={`direction-badge ${dir.cls}`}>
            {dir.text}
          </div>
        </div>
        <div className="pred-item">
          <div className="pred-label">Magnitude</div>
          <div className={`pred-value ${dir.cls}`}>{mag}%</div>
        </div>
        <div className="pred-item">
          <div className="pred-label">Confidence</div>
          <div className="pred-value neutral">{conf}%</div>
        </div>
      </div>

      <div className="prediction-grid" style={{ marginBottom: "1.25rem" }}>
        <div className="pred-item">
          <div className="pred-label">Regime</div>
          <div className="pred-value neutral">{regimeLabel(latest.regime)}</div>
        </div>
        <div className="pred-item">
          <div className="pred-label">Strategy Return</div>
          <div className={`pred-value ${latest.strategy_return > 0 ? "up" : latest.strategy_return < 0 ? "down" : "flat"}`}>
            {latest.strategy_return != null ? `${(latest.strategy_return * 100).toFixed(3)}%` : "—"}
          </div>
        </div>
        <div className="pred-item">
          <div className="pred-label">Model</div>
          <div className="pred-value neutral" style={{ fontSize: "0.85rem" }}>LightGBM</div>
        </div>
      </div>

      {/* Confidence trend chart */}
      {chartData.length > 2 && (
        <div className="chart-container" style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#64748b", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(148,163,184,0.08)" }}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  background: "#1e293b",
                  border: "1px solid rgba(148,163,184,0.15)",
                  borderRadius: "10px",
                  fontSize: "0.78rem",
                }}
              />
              <ReferenceLine y={50} stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
                name="Confidence %"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
