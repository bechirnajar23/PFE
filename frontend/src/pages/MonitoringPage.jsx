import React from "react";
import { Panel, SummaryCards } from "../components/Cards";
import { CpuMemoryChart, LatencyChart, WanChart } from "../components/Charts";

export default function MonitoringPage({ latest, prediction, counts, monitoring, loading }) {
  return (
    <>
      <SummaryCards latest={latest} prediction={prediction} counts={counts} loading={loading} />
      <section className="page-grid">
        <Panel title="CPU et memoire" action="Temps reel" className="full">
          <CpuMemoryChart data={monitoring} />
        </Panel>
        <Panel title="Latence reseau" action="ms">
          <LatencyChart data={monitoring} />
        </Panel>
        <Panel title="Debit WAN" action="KB/s">
          <WanChart data={monitoring} />
        </Panel>
      </section>
    </>
  );
}
