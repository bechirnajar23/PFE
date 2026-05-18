import React from "react";
import { AlertTriangle } from "lucide-react";
import { EmptyState } from "./Cards";
import { formatTime, prob } from "../utils/format";

export function EventList({ events }) {
  if (events.length === 0) return <EmptyState text="Aucune alerte critique recente" />;
  return (
    <div className="events-list">
      {events.map((event) => (
        <article key={`${event.timestamp}-${event.status_reason}`}>
          <strong>{event.local_status}</strong>
          <span>{event.alert_explanation || event.status_reason}</span>
          <em>{formatTime(event.timestamp)}</em>
        </article>
      ))}
    </div>
  );
}

export function PredictionRows({ rows }) {
  if (rows.length === 0) return <EmptyState text="Aucune prediction disponible" />;
  return (
    <div className="prediction-list">
      {rows.slice(0, 10).map((row) => (
        <article key={`${row.timestamp}-${row.horizon}`}>
          <strong>{row.horizon}</strong>
          <span>{prob(row.probability)}</span>
          <em>seuil {prob(row.threshold)}</em>
          <small>{row.diagnostic_dl || row.decision_message || "OK"}</small>
        </article>
      ))}
    </div>
  );
}

export function RulesGrid() {
  const rules = [
    ["URGENT", "Alerte email envoyee si l'etat courant HGW est urgent."],
    ["CRITICAL", "Email envoye si service critique arrete ou pression forte."],
    ["SHAP", "Explication locale des modeles CatBoost par contribution de features."],
  ];
  return (
    <div className="rules-grid">
      {rules.map(([title, text]) => (
        <article key={title}>
          <AlertTriangle size={19} />
          <strong>{title}</strong>
          <span>{text}</span>
        </article>
      ))}
    </div>
  );
}

function featureValue(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const number = Number(value);
  if (Number.isFinite(number)) return Math.round(number * 100) / 100;
  return value;
}

const IMPACT_LEVEL_LABEL = { fort: "Fort", "modéré": "Modéré", faible: "Faible" };

export function XaiReasons({ items }) {
  if (!items.length) {
    return <EmptyState text="SHAP en attente: lancez un cycle de prediction" />;
  }

  return (
    <div className="xai-list">
      {items.map((item) => {
        const shap = item.shap || item.explainer_json?.shap || {};
        const topFeatures = shap.top_features || [];
        const explanation = item.shap_explanation || shap.business_explanation;

        return (
          <article key={`${item.timestamp}-${item.horizon}`} className="xai-card">
            <header>
              <div>
                <strong>{item.horizon}</strong>
              </div>
              <em className={`risk-badge ${Number(item.probability) >= 0.7 ? "high" : Number(item.probability) >= 0.4 ? "medium" : "low"}`}>
                {prob(item.probability)}
              </em>
            </header>

            {explanation && <p className="xai-business">{explanation}</p>}

            <div className="xai-bars">
              {topFeatures.slice(0, 5).map((feature) => {
                const share = Math.max(0.03, Math.min(1, Number(feature.share || 0)));
                const aggravates = feature.impact !== "decrease";
                const level = feature.impact_level || "faible";
                return (
                  <div className="xai-row" key={`${item.horizon}-${feature.feature}`}>
                    <div className="xai-feature-info">
                      <strong>{feature.label || feature.feature}</strong>
                      <span>{feature.context || (feature.value != null ? `Valeur: ${featureValue(feature.value)}` : null)}</span>
                    </div>
                    <div className="xai-bar-track">
                      <span
                        className={aggravates ? "increase" : "decrease"}
                        style={{ width: `${Math.round(share * 100)}%` }}
                      />
                    </div>
                    <div className={`xai-badge ${aggravates ? "up" : "down"}`}>
                      <span>{aggravates ? "↑ Aggrave" : "↓ Protège"}</span>
                      <em>{IMPACT_LEVEL_LABEL[level] || level}</em>
                    </div>
                  </div>
                );
              })}
            </div>
          </article>
        );
      })}
    </div>
  );
}
