import React from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ComposedChart, Bar, Line,
} from "recharts";

/**
 * PriceChart – OHLCV area chart with volume bars
 * Data source: /api/prices/<asset> → market_pipeline.py → MongoDB prices collection
 */
export default function PriceChart({ data, asset }) {
  if (!data || data.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📈</div>
        <p>No price data available for {asset}</p>
      </div>
    );
  }

  const formatDate = (ts) => {
    if (!ts) return "";
    const d = new Date(ts);
    return d.toLocaleDateString("en-IN", { month: "short", day: "numeric" });
  };

  const formatPrice = (val) => {
    if (val == null) return "—";
    if (val >= 1000) return `₹${(val / 1000).toFixed(1)}K`;
    return `₹${val.toFixed(2)}`;
  };

  const prices = data.map((row) => ({
    date: formatDate(row.timestamp),
    close: row.close,
    high: row.high,
    low: row.low,
    open: row.open,
    volume: row.volume,
    fullDate: row.timestamp,
  }));

  const minClose = Math.min(...prices.map((p) => p.low || p.close).filter(Boolean)) * 0.998;
  const maxClose = Math.max(...prices.map((p) => p.high || p.close).filter(Boolean)) * 1.002;

  return (
    <div className="chart-container tall">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={prices} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(148,163,184,0.08)" }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="price"
            domain={[minClose, maxClose]}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickFormatter={formatPrice}
            tickLine={false}
            axisLine={false}
            width={65}
          />
          <YAxis
            yAxisId="vol"
            orientation="right"
            tick={false}
            axisLine={false}
            width={0}
          />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid rgba(148,163,184,0.15)",
              borderRadius: "10px",
              boxShadow: "0 10px 40px rgba(0,0,0,0.6)",
              fontSize: "0.78rem",
            }}
            labelStyle={{ color: "#f1f5f9", fontWeight: 700, marginBottom: 4 }}
            itemStyle={{ color: "#94a3b8" }}
            formatter={(value, name) => {
              if (name === "volume") return [value?.toLocaleString(), "Volume"];
              return [formatPrice(value), name.charAt(0).toUpperCase() + name.slice(1)];
            }}
          />
          <Bar
            yAxisId="vol"
            dataKey="volume"
            fill="url(#volGrad)"
            radius={[2, 2, 0, 0]}
            opacity={0.5}
          />
          <Area
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            strokeWidth={2.5}
            fill="url(#priceGrad)"
            dot={false}
            activeDot={{ r: 5, fill: "#3b82f6", stroke: "#fff", strokeWidth: 2 }}
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="high"
            stroke="#10b981"
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            opacity={0.5}
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="low"
            stroke="#f43f5e"
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            opacity={0.5}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
