import React from "react";
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar,
} from "recharts";

/**
 * TechnicalsPanel – Displays technical indicator charts from features collection
 * Data source: /api/features/<asset>
 *   → cleaning_pipeline.py → engineer_price_features() computes RSI, ATR, volatility
 *   → add_cross_features() computes sentiment_x_returns, event_x_volatility
 *
 * Shows: RSI line, Rolling Volatility, Volume Z-Score, Sentiment trend
 */
export default function TechnicalsPanel({ features, loading }) {
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading technicals...</span>
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <p>No feature data available. Run cleaning_pipeline.py first.</p>
      </div>
    );
  }

  const fmt = (ts) => {
    if (!ts) return "";
    return new Date(ts).toLocaleDateString("en-IN", { month: "short", day: "numeric" });
  };

  const data = features.map((row) => ({
    date: fmt(row.timestamp),
    rsi: row.rsi != null ? +Number(row.rsi).toFixed(1) : null,
    volatility: row.rolling_volatility != null ? +(row.rolling_volatility * 100).toFixed(3) : null,
    volZscore: row.volume_zscore != null ? +Number(row.volume_zscore).toFixed(2) : null,
    sentiment: row.avg_sentiment != null ? +Number(row.avg_sentiment).toFixed(4) : null,
    atr: row.atr != null ? +Number(row.atr).toFixed(2) : null,
  }));

  const chartStyle = {
    background: "#1e293b",
    border: "1px solid rgba(148,163,184,0.15)",
    borderRadius: "10px",
    fontSize: "0.75rem",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
      {/* RSI */}
      <div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>
          RSI (14)
        </div>
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(148,163,184,0.08)" }} />
              <YAxis domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} width={30} />
              <Tooltip contentStyle={chartStyle} />
              <ReferenceLine y={70} stroke="rgba(244,63,94,0.3)" strokeDasharray="4 4" label={{ value: "70", fill: "#f43f5e", fontSize: 9 }} />
              <ReferenceLine y={30} stroke="rgba(16,185,129,0.3)" strokeDasharray="4 4" label={{ value: "30", fill: "#10b981", fontSize: 9 }} />
              <Line type="monotone" dataKey="rsi" stroke="#f59e0b" strokeWidth={2} dot={false} name="RSI" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Rolling Volatility */}
      <div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>
          Rolling Volatility (10d %)
        </div>
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="volGrad2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(148,163,184,0.08)" }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} width={35} />
              <Tooltip contentStyle={chartStyle} />
              <Area type="monotone" dataKey="volatility" stroke="#f43f5e" strokeWidth={2} fill="url(#volGrad2)" dot={false} name="Volatility %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sentiment */}
      <div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>
          Avg Sentiment (VADER)
        </div>
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(148,163,184,0.08)" }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} width={40} />
              <Tooltip contentStyle={chartStyle} />
              <ReferenceLine y={0} stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
              <Area type="monotone" dataKey="sentiment" stroke="#10b981" strokeWidth={2} fill="url(#sentGrad)" dot={false} name="Sentiment" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Volume Z-Score */}
      <div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>
          Volume Z-Score
        </div>
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(148,163,184,0.08)" }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} width={30} />
              <Tooltip contentStyle={chartStyle} />
              <ReferenceLine y={0} stroke="rgba(148,163,184,0.2)" />
              <ReferenceLine y={2} stroke="rgba(244,63,94,0.3)" strokeDasharray="4 4" />
              <ReferenceLine y={-2} stroke="rgba(16,185,129,0.3)" strokeDasharray="4 4" />
              <Bar dataKey="volZscore" name="Vol Z-Score" radius={[2, 2, 0, 0]}>
                {data.map((entry, index) => {
                  const color = (entry.volZscore || 0) > 2
                    ? "#f43f5e"
                    : (entry.volZscore || 0) < -2
                    ? "#10b981"
                    : "#6366f1";
                  return <rect key={index} fill={color} fillOpacity={0.6} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
