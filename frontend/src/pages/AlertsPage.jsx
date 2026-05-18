import React from "react";
import { Panel, SummaryCards } from "../components/Cards";
import { StateDonut } from "../components/Charts";
import { EventList, RulesGrid, XaiReasons } from "../components/Lists";

export default function AlertsPage({ latest, prediction, counts, events, xai, loading }) {
  const stable = Math.max(1, Number(counts.snapshots || 0) - Number(counts.critical_last_hour || 0));
  const critical = Number(counts.critical_last_hour || 0);

  return (
    <>
      <SummaryCards latest={latest} prediction={prediction} counts={counts} loading={loading} />
      <section className="page-grid">
        <Panel title="Evenements critiques" action="URGENT / CRITICAL" className="full">
          <EventList events={events} />
        </Panel>
        <Panel title="Repartition des etats" action="Derniere heure">
          <StateDonut stable={stable} critical={critical} />
        </Panel>
        <Panel title="Regles d'alerte" action="Email">
          <RulesGrid />
        </Panel>
        <Panel title="Pourquoi le risque monte" action="SHAP" className="full">
          <XaiReasons items={xai} />
        </Panel>
      </section>
    </>
  );
}
