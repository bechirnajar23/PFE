import { useEffect, useState } from "react";
import { formatTime, normalizeHorizon } from "../utils/format";

const API_BASE = import.meta.env.VITE_API_URL || "";
const refreshMs = Number(import.meta.env.VITE_REFRESH_MS || 5000);
const REFRESH_MS = Number.isFinite(refreshMs) && refreshMs >= 1000 ? refreshMs : 5000;
const FETCH_OPTIONS = { cache: "no-store" };

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function buildPredictionChart(items) {
  const byTimestamp = new Map();
  for (const item of items) {
    const key = item.timestamp;
    const horizon = normalizeHorizon(item.horizon);
    const row = byTimestamp.get(key) || { timestamp: key, time: formatTime(key) };
    row[horizon] = Number(item.probability || 0);
    row[`${horizon} seuil`] = Number(item.threshold || 0);
    byTimestamp.set(key, row);
  }
  return Array.from(byTimestamp.values());
}

export function useDashboardData() {
  const [data, setData] = useState({
    summary: null,
    monitoring: [],
    predictions: [],
    predictionRows: [],
    events: [],
    xai: [],
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [summaryRes, monitoringRes, predictionSeriesRes, predictionRowsRes, eventRes, xaiRes] = await Promise.all([
          fetch(apiUrl("/api/summary"), FETCH_OPTIONS),
          fetch(apiUrl("/api/series/monitoring?limit=260"), FETCH_OPTIONS),
          fetch(apiUrl("/api/series/predictions?limit=700"), FETCH_OPTIONS),
          fetch(apiUrl("/api/predictions/latest?limit=18"), FETCH_OPTIONS),
          fetch(apiUrl("/api/events/recent?limit=8"), FETCH_OPTIONS),
          fetch(apiUrl("/api/xai/latest?limit=8"), FETCH_OPTIONS),
        ]);

        if (!summaryRes.ok) throw new Error(`Backend API HTTP ${summaryRes.status}`);
        const [summary, monitoringPayload, predictionSeriesPayload, predictionRowsPayload, eventPayload, xaiPayload] = await Promise.all([
          summaryRes.json(),
          monitoringRes.ok ? monitoringRes.json() : { items: [] },
          predictionSeriesRes.ok ? predictionSeriesRes.json() : { items: [] },
          predictionRowsRes.ok ? predictionRowsRes.json() : { items: [] },
          eventRes.ok ? eventRes.json() : { items: [] },
          xaiRes.ok ? xaiRes.json() : { items: [] },
        ]);

        if (cancelled) return;
        setData({
          summary,
          monitoring: (monitoringPayload.items || []).map((item) => ({
            ...item,
            time: formatTime(item.timestamp),
            cpu_usage_percent: Number(item.cpu_usage_percent || 0),
            mem_usage_percent: Number(item.mem_usage_percent || 0),
            net_latency_ms: Number(item.net_latency_ms || 0),
            wan_rx_rate_kbps: Number(item.wan_rx_rate_kbps || 0),
            wan_tx_rate_kbps: Number(item.wan_tx_rate_kbps || 0),
          })),
          predictions: buildPredictionChart(predictionSeriesPayload.items || []),
          predictionRows: predictionRowsPayload.items || [],
          events: eventPayload.items || [],
          xai: xaiPayload.items || [],
          loading: false,
          error: null,
        });
      } catch (error) {
        if (!cancelled) {
          setData((current) => ({
            ...current,
            loading: false,
            error: error.message || "Backend indisponible",
          }));
        }
      }
    }

    load();
    const timer = window.setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return data;
}
