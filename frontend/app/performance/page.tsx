"use client";
import { useEffect, useState } from "react";
import { Download, RefreshCw } from "lucide-react";
import { api, PageHeading } from "../../components/workbench";
type Metrics = {
  num_samples: number;
  age: { mae: number; rmse: number; median_ae: number };
  gender: { accuracy: number; macro_f1: number; confusion_matrix: number[][] };
  composite_score: number;
};
type Report = { id: string; metrics: Metrics };
export default function Page() {
  const [reports, setReports] = useState<Report[]>([]),
    [selected, setSelected] = useState(""),
    [note, setNote] = useState(""),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true);
  const load = () => {
    setLoading(true);
    setError("");
    api<{ reports: Report[]; note: string }>("evaluation")
      .then((d) => {
        setReports(d.reports);
        setSelected(
          d.reports.find((r) => r.id.includes("final"))?.id ||
            d.reports[0]?.id ||
            "",
        );
        setNote(d.note);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);
  const report = reports.find((r) => r.id === selected);
  const m = report?.metrics;
  const matrix = m?.gender.confusion_matrix;
  const max = matrix ? Math.max(...matrix.flat()) : 1;
  return (
    <>
      <PageHeading
        eyebrow="MEASURE, THEN UNDERSTAND"
        title={
          <>
            Evidence over <em>assumptions.</em>
          </>
        }
        description="Explore recorded evaluation runs, age errors, and classification outcomes."
      />
      <div className="notice">
        <span>↗</span>
        <p>
          <strong>Historical evaluation archive.</strong>{" "}
          {note ||
            "These saved reports predate the checkpoint output-layer correction. A fresh evaluation is needed to measure the corrected model."}
        </p>
      </div>
      <div className="toolbar">
        <label>
          Evaluation run{" "}
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            disabled={loading}
          >
            {reports.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <button className="button secondary" onClick={load} disabled={loading}>
          <RefreshCw size={14} className={loading ? "spin" : ""} /> Refresh
          archive
        </button>
      </div>
      {error && (
        <div role="alert" className="error-message">
          {error}
        </div>
      )}
      {loading && <p role="status">Loading evaluation reports…</p>}
      {!loading && !error && !m && (
        <p>No saved evaluation reports are available.</p>
      )}
      {m && (
        <>
          <div className="metric-grid">
            {[
              {
                label: "AGE RMSE",
                value: m.age.rmse.toFixed(2),
                unit: "years",
                text: "Root mean squared age error",
              },
              {
                label: "GENDER MACRO F1",
                value: (m.gender.macro_f1 * 100).toFixed(1),
                unit: "%",
                text: "Balanced across both dataset labels",
              },
              {
                label: "LABEL ACCURACY",
                value: (m.gender.accuracy * 100).toFixed(1),
                unit: "%",
                text: "Correct predictions in this run",
              },
              {
                label: "EVALUATED IMAGES",
                value: m.num_samples.toLocaleString(),
                unit: "",
                text: "Sample size of the saved run",
              },
            ].map((x) => (
              <div className="metric-card" key={x.label}>
                <span className="eyebrow">{x.label}</span>
                <strong>
                  {x.value}
                  <small>{x.unit}</small>
                </strong>
                <p>{x.text}</p>
              </div>
            ))}
          </div>
          <div className="two-column">
            <section className="panel chart-panel">
              <div className="panel-heading">
                Age estimation errors{" "}
                <span className="mono">YEARS · LOWER IS BETTER</span>
              </div>
              <div className="bar-chart">
                {[
                  { label: "Mean absolute error", value: m.age.mae },
                  { label: "Root mean squared error", value: m.age.rmse },
                  { label: "Median absolute error", value: m.age.median_ae },
                ].map((x) => (
                  <div key={x.label}>
                    <div className="bar-label">
                      <span>{x.label}</span>
                      <strong>{x.value.toFixed(2)} years</strong>
                    </div>
                    <div className="bar-track">
                      <div style={{ width: `${Math.min(100, x.value)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <p className="small-note">
                Bars share a 0–100 year scale. RMSE gives greater weight to
                large errors; MAE describes average absolute error.
              </p>
            </section>
            <section className="panel chart-panel">
              <div className="panel-heading">
                Classification outcomes{" "}
                <span className="mono">CONFUSION MATRIX</span>
              </div>
              <div className="matrix-wrap">
                <div className="matrix-axis">Predicted label →</div>
                <div className="matrix">
                  <span />
                  <span>Female</span>
                  <span>Male</span>
                  {matrix?.map((row, i) => (
                    <div className="matrix-row" key={i}>
                      <span>Actual {i === 0 ? "Female" : "Male"}</span>
                      {row.map((v, j) => (
                        <div
                          className={`matrix-cell ${i === j ? "correct" : ""}`}
                          style={{
                            background:
                              i === j
                                ? `rgba(41,107,109,${0.12 + (v / max) * 0.25})`
                                : `rgba(184,111,76,${0.07 + (v / max) * 0.23})`,
                          }}
                          key={j}
                        >
                          <strong>{v}</strong>
                          <small>{i === j ? "Correct" : "Misclassified"}</small>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
              <p className="small-note">
                Rows show actual dataset labels; columns show predictions.
                Diagonal cells are correct classifications.
              </p>
            </section>
          </div>
          <div className="panel report-foot">
            <div>
              <h3>Keep the source close.</h3>
              <p>
                Composite score: <strong>{m.composite_score.toFixed(4)}</strong>{" "}
                · Saved report: <span className="mono">{selected}</span>
              </p>
            </div>
            <a
              className="button secondary"
              href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(report, null, 2))}`}
              download={`${selected}.json`}
            >
              <Download size={15} /> Download report
            </a>
          </div>
        </>
      )}
    </>
  );
}
