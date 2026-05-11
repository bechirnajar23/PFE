import React from "react";
import { Panel, SummaryCards } from "../components/Cards";
import { CpuMemoryChart, RiskChart, StateDonut, WanChart } from "../components/Charts";
import { EventList } from "../components/Lists";

export default function HomePage({ latest, prediction, counts, monitoring, predictions, events, loading }) {
  const stable = Math.max(1, Number(counts.snapshots || 0) - Number(counts.critical_last_hour || 0));
  const critical = Number(counts.critical_last_hour || 0);

  return (
    <>
      <SummaryCards latest={latest} prediction={prediction} counts={counts} loading={loading} />
      <section className="dashboard-grid">
        <Panel title="CPU et memoire" action="monitor_snapshots" className="wide">
          <CpuMemoryChart data={monitoring} />
        </Panel>
        <Panel title="Risque multi-horizon" action="predictions_log">
          <RiskChart data={predictions} />
        </Panel>
        <Panel title="Debit WAN" action="RX / TX">
          <WanChart data={monitoring} />
        </Panel>
        <Panel title="Repartition des etats" action="Derniere heure">
          <StateDonut stable={stable} critical={critical} />
        </Panel>
        <Panel title="Evenements recents" action="Alertes metier">
          <EventList events={events} />
        </Panel>
      </section>
    </>
  );
}
