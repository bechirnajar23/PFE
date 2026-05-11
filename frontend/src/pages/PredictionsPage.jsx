import React from "react";
import { Panel, SummaryCards } from "../components/Cards";
import { RiskChart } from "../components/Charts";
import { PredictionRows, XaiReasons } from "../components/Lists";

export default function PredictionsPage({ latest, prediction, counts, predictions, predictionRows, xai, loading }) {
  return (
    <>
      <SummaryCards latest={latest} prediction={prediction} counts={counts} loading={loading} />
      <section className="page-grid">
        <Panel title="Probabilites multi-horizon" action="CatBoost + LSTM" className="full">
          <RiskChart data={predictions} />
        </Panel>
        <Panel title="Seuils de decision" action="probabilite >= seuil">
          <RiskChart data={predictions} thresholds />
        </Panel>
        <Panel title="Dernieres predictions" action="diagnostic DL">
          <PredictionRows rows={predictionRows} />
        </Panel>
        <Panel title="Explication SHAP" action="XAI CatBoost" className="full">
          <XaiReasons items={xai} />
        </Panel>
      </section>
    </>
  );
}
