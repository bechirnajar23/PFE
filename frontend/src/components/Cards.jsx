import React from "react";
import { BrainCircuit, Cpu, Router, ShieldAlert } from "lucide-react";
import { ms, pct, prob } from "../utils/format";

export function MetricCard({ title, value, detail, icon: Icon, tone = "green" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <div>
        <span>{title}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
      <Icon size={24} />
    </article>
  );
}

export function SmallCard({ title, value, detail, icon: Icon }) {
  return (
    <article className="small-card">
      <Icon size={21} />
      <div>
        <strong>{title}</strong>
        <span>{value}</span>
      </div>
      <em>{detail}</em>
    </article>
  );
}

export function Panel({ title, action, className = "", children }) {
  return (
    <section className={`panel ${className}`}>
      <header>
        <h2>{title}</h2>
        {action ? <span>{action}</span> : null}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function EmptyState({ text = "En attente de donnees" }) {
  return <div className="empty-state">{text}</div>;
}

export function SummaryCards({ latest, prediction, counts, loading }) {
  const status = latest?.local_status;
  const isUrgent = status === "URGENT";
  const isCritical = status === "CRITICAL";
  const isWarning = status === "WARNING";
  const statusTone = isCritical ? "alert" : isUrgent ? "urgent" : isWarning ? "warning" : "green";
  return (
    <section className="summary-grid">
      <MetricCard
        icon={Router}
        title="Etat HGW"
        value={loading ? "..." : latest?.local_status || "En attente"}
        detail={latest?.status_reason || "Dernier statut metier"}
        tone={statusTone}
      />
      <MetricCard
        icon={Cpu}
        title="CPU / Memoire"
        value={`${pct(latest?.cpu_usage_percent)} | ${pct(latest?.mem_usage_percent)}`}
        detail={`Latence ${ms(latest?.net_latency_ms)}`}
        tone="navy"
      />
      <div className="small-stack">
        <SmallCard icon={BrainCircuit} title="Risque max" value={prob(prediction?.max_probability)} detail={prediction?.decision_level || "OK"} />
        <SmallCard icon={ShieldAlert} title="Alertes 1h" value={String(counts.critical_last_hour ?? 0)} detail="URGENT / CRITICAL" />
      </div>
    </section>
  );
}
