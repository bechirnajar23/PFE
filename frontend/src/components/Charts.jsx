import React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EmptyState } from "./Cards";

export const horizonColors = {
  "15min": "#77b65d",
  "30min": "#243047",
  "60min": "#34b3a0",
  "360min": "#88a2c4",
  "3 jours": "#e0607e",
};

export function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {payload.map((item) => (
        <span key={item.dataKey}>
          <i style={{ background: item.color }} />
          {item.name}: {Number(item.value).toFixed(item.value <= 1 ? 3 : 1)}
        </span>
      ))}
    </div>
  );
}

export function CpuMemoryChart({ data }) {
  if (data.length === 0) return <EmptyState />;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id="cpu" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#77b65d" stopOpacity={0.36} />
            <stop offset="95%" stopColor="#77b65d" stopOpacity={0.04} />
          </linearGradient>
          <linearGradient id="mem" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#243047" stopOpacity={0.26} />
            <stop offset="95%" stopColor="#243047" stopOpacity={0.03} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(36,48,71,0.12)" vertical={false} />
        <XAxis dataKey="time" tickLine={false} axisLine={false} stroke="#69758a" />
        <YAxis tickLine={false} axisLine={false} stroke="#69758a" unit="%" />
        <Tooltip content={<ChartTooltip />} />
        <Legend />
        <Area dataKey="cpu_usage_percent" name="CPU %" stroke="#77b65d" fill="url(#cpu)" strokeWidth={2.7} />
        <Area dataKey="mem_usage_percent" name="Memoire %" stroke="#243047" fill="url(#mem)" strokeWidth={2.7} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function LatencyChart({ data }) {
  if (data.length === 0) return <EmptyState />;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid stroke="rgba(36,48,71,0.12)" vertical={false} />
        <XAxis dataKey="time" tickLine={false} axisLine={false} stroke="#69758a" />
        <YAxis tickLine={false} axisLine={false} stroke="#69758a" />
        <Tooltip content={<ChartTooltip />} />
        <Line dataKey="net_latency_ms" name="Latence ms" stroke="#34b3a0" strokeWidth={2.7} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function WanChart({ data }) {
  if (data.length === 0) return <EmptyState />;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data.slice(-18)}>
        <CartesianGrid stroke="rgba(36,48,71,0.12)" vertical={false} />
        <XAxis dataKey="time" tickLine={false} axisLine={false} stroke="#69758a" />
        <YAxis tickLine={false} axisLine={false} stroke="#69758a" />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="wan_rx_rate_kbps" name="WAN RX" fill="#77b65d" radius={[6, 6, 0, 0]} />
        <Bar dataKey="wan_tx_rate_kbps" name="WAN TX" fill="#243047" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RiskChart({ data, thresholds = false }) {
  if (data.length === 0) return <EmptyState />;
  const keys = ["15min", "30min", "60min", "360min", "3 jours"];
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid stroke="rgba(36,48,71,0.12)" vertical={false} />
        <XAxis dataKey="time" tickLine={false} axisLine={false} stroke="#69758a" />
        <YAxis
          tickLine={false}
          axisLine={false}
          stroke="#69758a"
          domain={[0, 1]}
          tickFormatter={(value) => `${Math.round(value * 100)}%`}
        />
        <Tooltip content={<ChartTooltip />} />
        <Legend />
        {keys.map((key) => (
          <Line
            key={key}
            dataKey={thresholds ? `${key} seuil` : key}
            name={thresholds ? `${key} seuil` : key}
            stroke={horizonColors[key]}
            strokeDasharray={thresholds ? "6 5" : undefined}
            strokeWidth={thresholds ? 2 : 2.6}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function StateDonut({ stable, critical }) {
  const data = [
    { name: "Stable", value: Math.max(1, stable) },
    { name: "Critique", value: Math.max(0, critical) },
  ];

  return (
    <div className="donut-layout">
      <ResponsiveContainer width="100%" height={210}>
        <PieChart>
          <Pie data={data} innerRadius={58} outerRadius={86} paddingAngle={4} dataKey="value">
            <Cell fill="#77b65d" />
            <Cell fill="#e0607e" />
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div className="donut-legend">
        <span><i className="stable" />Stable</span>
        <span><i className="critical" />Critique</span>
      </div>
    </div>
  );
}
