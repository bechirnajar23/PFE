import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bell, BrainCircuit, Gauge, Home } from "lucide-react";
import HomePage from "./pages/HomePage";
import MonitoringPage from "./pages/MonitoringPage";
import PredictionsPage from "./pages/PredictionsPage";
import AlertsPage from "./pages/AlertsPage";
import { useDashboardData } from "./hooks/useDashboardData";
import "./styles.css";

const navItems = [
  { key: "home", label: "Accueil", icon: Home },
  { key: "monitoring", label: "Monitoring", icon: Gauge },
  { key: "predictions", label: "Predictions", icon: BrainCircuit },
  { key: "alerts", label: "Alertes", icon: Bell },
];

function App() {
  const [activePage, setActivePage] = useState("home");
  const { summary, monitoring, predictions, predictionRows, events, xai, loading, error } = useDashboardData();
  const latest = summary?.latestSnapshot;
  const prediction = summary?.latestPrediction;
  const services = summary?.services || {};
  const counts = summary?.counts || {};

  const pageTitle = {
    home: "Dashboard de supervision",
    monitoring: "Monitoring operationnel",
    predictions: "Predictions IA",
    alerts: "Alertes et diagnostic",
  }[activePage];

  const now = useMemo(
    () =>
      new Intl.DateTimeFormat("fr-FR", {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date()),
    [summary],
  );

  const commonProps = {
    latest,
    prediction,
    counts,
    monitoring,
    predictions,
    predictionRows,
    events,
    xai,
    loading,
  };

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <strong>HGW Admin</strong>
          </div>
        </div>

        <nav className="nav" aria-label="Navigation principale">
          {navItems.map(({ key, label, icon: Icon }) => (
            <button
              className={activePage === key ? "active" : ""}
              type="button"
              key={key}
              onClick={() => setActivePage(key)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-section">
          <span>Services</span>
          {[
            ["API", services.api?.status || (error ? "DOWN" : "UP")],
            ["DB", services.database?.status || "UP"],
            ["Grafana", services.grafana?.status || "UP"],
          ].map(([name, status]) => (
            <div className="service-row" key={name}>
              <i className={status === "UP" ? "ok" : "warn"} />
              <strong>{name}</strong>
              <em>{status}</em>
            </div>
          ))}
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <div>
            <span>HGW Predictive System</span>
            <h1>{pageTitle}</h1>
          </div>
          <div className="topbar-right">
            <span>{now}</span>
          </div>
        </header>

        {error ? <div className="api-warning">Backend API indisponible: {error}</div> : null}

        {activePage === "home" && <HomePage {...commonProps} />}
        {activePage === "monitoring" && <MonitoringPage {...commonProps} />}
        {activePage === "predictions" && <PredictionsPage {...commonProps} />}
        {activePage === "alerts" && <AlertsPage {...commonProps} />}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
