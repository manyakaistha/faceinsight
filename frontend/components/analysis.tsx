"use client";
import { useEffect, useRef, useState } from "react";
import {
  Upload,
  ArrowUpRight,
  LoaderCircle,
  Download,
  RotateCcw,
  ImagePlus,
  Check,
  ScanFace,
} from "lucide-react";
import { api, PageHeading } from "./workbench";
import Field from "./field";
type Face = {
  bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
  estimated_age: number;
  predicted_label: string;
  gender_probability: number;
  decision_threshold: number;
};
type Prediction = {
  success: boolean;
  faces: Face[];
  model_version: string;
  device: string;
  error_message?: string;
};
type Explanation = {
  success: boolean;
  target: string;
  original: string;
  heatmap: string;
  overlay: string;
  model_version: string;
};
export default function Analysis({ explain = false }: { explain?: boolean }) {
  const [file, setFile] = useState<File | null>(null),
    [url, setUrl] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [drag, setDrag] = useState(false),
    [prediction, setPrediction] = useState<Prediction | null>(null),
    [explanation, setExplanation] = useState<Explanation | null>(null),
    [target, setTarget] = useState("gender"),
    [view, setView] = useState("overlay"),
    [dimensions, setDimensions] = useState({ w: 1, h: 1 }),
    [elapsed, setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!file) {
      setUrl("");
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  const select = (f: File) => {
    if (busy) return;
    setError("");
    setPrediction(null);
    setExplanation(null);
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("Choose a JPG, PNG, or WebP image.");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("This image exceeds 10 MB. Choose a smaller image.");
      return;
    }
    setFile(f);
  };
  const sample = async (id: number) => {
    setError("");
    try {
      const r = await fetch(`/api/v1/samples/portrait-${id}`);
      if (!r.ok) throw new Error("Sample image unavailable.");
      select(
        new File([await r.blob()], `sample-portrait-${id}.jpg`, {
          type: "image/jpeg",
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const run = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setPrediction(null);
    setExplanation(null);
    const start = performance.now();
    const form = new FormData();
    form.append("file", file);
    try {
      if (explain) {
        setExplanation(
          await api<Explanation>(`explain?target=${target}`, {
            method: "POST",
            body: form,
          }),
        );
        setView("overlay");
      } else {
        const r = await api<Prediction>("predict", {
          method: "POST",
          body: form,
        });
        if (!r.success)
          throw new Error(
            r.error_message ||
              "No face detected. Try a clear front-facing portrait.",
          );
        setPrediction(r);
      }
      setElapsed((performance.now() - start) / 1000);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const reset = () => {
    setFile(null);
    setPrediction(null);
    setExplanation(null);
    setError("");
    if (input.current) input.current.value = "";
  };
  const download = () => {
    if (!prediction) return;
    const a = document.createElement("a");
    const u = URL.createObjectURL(
      new Blob([JSON.stringify(prediction, null, 2)], {
        type: "application/json",
      }),
    );
    a.href = u;
    a.download = "faceinsight-prediction.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(u), 1000);
  };
  return (
    <>
      <PageHeading
        eyebrow={explain ? "FOLLOW THE ATTENTION" : "A PORTRAIT, A PERSPECTIVE"}
        title={
          explain ? (
            <>
              Behind the <em>prediction.</em>
            </>
          ) : (
            <>
              Look a little <em>closer.</em>
            </>
          )
        }
        description={
          explain
            ? "Generate a live Grad-CAM explanation and explore which regions influence the model."
            : "Upload a portrait and explore real age estimates and dataset-label predictions."
        }
      />
      <div className="analysis-grid">
        <section className="panel image-panel">
          <div className="panel-heading">
            <span>
              <ImagePlus size={16} />{" "}
              {explain ? "Source portrait" : "Image workspace"}
            </span>
            <span className="mono">01 / INPUT</span>
          </div>
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            aria-label="Upload a portrait"
            className="sr-only"
            onChange={(e) => {
              if (e.target.files?.[0]) select(e.target.files[0]);
            }}
            disabled={busy}
          />
          {file ? (
            <div className="image-stage">
              <div className="image-wrap">
                <img
                  src={url}
                  alt="Selected portrait"
                  onLoad={(e) =>
                    setDimensions({
                      w: e.currentTarget.naturalWidth,
                      h: e.currentTarget.naturalHeight,
                    })
                  }
                />
                {prediction?.faces.map((f, i) => (
                  <div
                    key={i}
                    className="face-box"
                    style={{
                      left: `${(f.bbox.x1 / dimensions.w) * 100}%`,
                      top: `${(f.bbox.y1 / dimensions.h) * 100}%`,
                      width: `${((f.bbox.x2 - f.bbox.x1) / dimensions.w) * 100}%`,
                      height: `${((f.bbox.y2 - f.bbox.y1) / dimensions.h) * 100}%`,
                    }}
                  >
                    <span>FACE {String(i + 1).padStart(2, "0")}</span>
                  </div>
                ))}
              </div>
              <div className="file-info">
                <span>{file.name}</span>
                <span>{(file.size / 1024).toFixed(0)} KB</span>
                <button
                  className="text-link"
                  disabled={busy}
                  onClick={() => input.current?.click()}
                >
                  Change image
                </button>
              </div>
            </div>
          ) : (
            <button
              className={`drop-zone ${drag ? "dragging" : ""}`}
              onClick={() => input.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                if (e.dataTransfer.files[0]) select(e.dataTransfer.files[0]);
              }}
            >
              <span className="upload-orbit">
                <Upload size={26} />
              </span>
              <strong>Give the model a new perspective</strong>
              <span>Drop a portrait here, or click to browse</span>
              <small>JPG, PNG, WEBP · up to 10 MB</small>
            </button>
          )}
          <div className="sample-area">
            <div className="eyebrow">OR EXPLORE A SAMPLE</div>
            <div className="sample-grid">
              {[1, 2, 3].map((n) => (
                <button
                  key={n}
                  className="sample-card"
                  disabled={busy}
                  onClick={() => sample(n)}
                >
                  <img
                    src={`/api/v1/samples/portrait-${n}`}
                    alt={`Sample portrait ${n}`}
                  />
                  <span>
                    Portrait 0{n} <ArrowUpRight size={13} />
                  </span>
                </button>
              ))}
            </div>
          </div>
          <div className="run-bar">
            {explain && (
              <label className="target-control">
                Explain{" "}
                <select
                  value={target}
                  onChange={(e) => {
                    setTarget(e.target.value);
                    setExplanation(null);
                  }}
                  disabled={busy}
                >
                  <option value="gender">Dataset label</option>
                  <option value="age">Age estimate</option>
                </select>
              </label>
            )}
            <button
              className="button primary"
              onClick={run}
              disabled={!file || busy}
            >
              {busy ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <ScanFace size={17} />
              )}{" "}
              {busy
                ? "Exploring features…"
                : explain
                  ? "Generate explanation"
                  : "Analyze portrait"}
              {!busy && <ArrowUpRight size={16} />}
            </button>
            {file && (
              <button
                className="icon-button"
                aria-label="Clear image and results"
                onClick={reset}
                disabled={busy}
              >
                <RotateCcw size={17} />
              </button>
            )}
          </div>
        </section>
        <section className="panel result-panel">
          <div className="panel-heading">
            <span>{explain ? "Attention map" : "Prediction desk"}</span>
            <span className="mono">02 / {explain ? "EXPLAIN" : "RESULTS"}</span>
          </div>
          <div aria-live="polite">
            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}
            {busy ? (
              <div className="empty-result">
                <Field compact />
                <h2>
                  {explain
                    ? "Following the attention…"
                    : "Reading the features…"}
                </h2>
                <p>The local model is processing your portrait.</p>
              </div>
            ) : prediction ? (
              <div className="results">
                <div className="result-meta">
                  <span>
                    <Check size={13} /> {prediction.faces.length} face
                    {prediction.faces.length !== 1 ? "s" : ""} detected
                  </span>
                  <span>
                    {elapsed.toFixed(2)}s · {prediction.device.toUpperCase()}
                  </span>
                </div>
                {prediction.faces.map((f, i) => (
                  <article className="face-result" key={i}>
                    <div className="eyebrow">
                      FACE {String(i + 1).padStart(2, "0")}
                    </div>
                    <div className="prediction-numbers">
                      <div>
                        <span>Estimated age</span>
                        <strong>
                          {f.estimated_age.toFixed(1)}
                          <small>years</small>
                        </strong>
                      </div>
                      <div>
                        <span>Predicted dataset label</span>
                        <strong className="label-value">
                          {f.predicted_label}
                        </strong>
                      </div>
                    </div>
                    <div className="probability-label">
                      <span>Male-label probability</span>
                      <strong>
                        {(f.gender_probability * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div className="probability-track">
                      <div
                        style={{ width: `${f.gender_probability * 100}%` }}
                      />
                      <i style={{ left: `${f.decision_threshold * 100}%` }} />
                    </div>
                    <div className="probability-ends">
                      <span>Female</span>
                      <span>Threshold {f.decision_threshold.toFixed(2)}</span>
                      <span>Male</span>
                    </div>
                  </article>
                ))}
                <p className="small-note">
                  This is a model estimate, not a determination of identity.
                  Probability is an output score, not a calibrated confidence
                  measure.
                </p>
                <button className="button secondary" onClick={download}>
                  <Download size={15} /> Download results
                </button>
              </div>
            ) : explanation ? (
              <div className="explanation-result">
                <div className="result-meta">
                  <span>
                    <Check size={13} />{" "}
                    {explanation.target === "age" ? "Age" : "Dataset-label"}{" "}
                    attention
                  </span>
                  <span>{elapsed.toFixed(2)}s</span>
                </div>
                <div
                  className="segmented"
                  role="group"
                  aria-label="Explanation image view"
                >
                  {["original", "heatmap", "overlay"].map((v) => (
                    <button
                      key={v}
                      className={view === v ? "selected" : ""}
                      onClick={() => setView(v)}
                    >
                      {v}
                    </button>
                  ))}
                </div>
                <img
                  className="heatmap-image"
                  src={explanation[view as "original" | "heatmap" | "overlay"]}
                  alt={`${view} for the selected portrait`}
                />
                <div className="heatmap-scale" />
                <div className="probability-ends">
                  <span>Lower activation</span>
                  <span>Higher activation</span>
                </div>
                <p className="small-note">
                  Warmer regions contribute more to this output in the final
                  convolutional layer. Grad-CAM shows relative activation, not a
                  causal explanation. The largest detected face is shown.
                </p>
                <a
                  className="button secondary"
                  href={explanation[view as "original" | "heatmap" | "overlay"]}
                  download={`faceinsight-${view}.png`}
                >
                  <Download size={15} /> Save {view}
                </a>
              </div>
            ) : (
              <div className="empty-result">
                <Field compact />
                <h2>
                  {explain
                    ? "Make the invisible visible."
                    : "A fresh point of view."}
                </h2>
                <p>
                  {explain
                    ? "Choose a portrait and an output to reveal the model’s attention."
                    : "Your analysis will appear here. Start with a portrait or choose a sample."}
                </p>
                <span className="empty-tag">
                  {explain ? "LIVE GRAD-CAM" : "REAL MODEL · LOCAL INFERENCE"}
                </span>
              </div>
            )}
          </div>
        </section>
      </div>
      <div className="guidance-grid">
        <div>
          <span>01</span>
          <h3>Keep it clear</h3>
          <p>
            A well-lit, front-facing portrait gives the detector the best chance
            to find a face.
          </p>
        </div>
        <div>
          <span>02</span>
          <h3>{explain ? "Follow the color" : "Explore multiple faces"}</h3>
          <p>
            {explain
              ? "Compare the source, heatmap, and overlay to inspect relative activation."
              : "The model analyzes each detected face and marks its location on your image."}
          </p>
        </div>
        <div>
          <span>03</span>
          <h3>Stay curious</h3>
          <p>
            Use the results as a starting point. Review the evaluation evidence
            and model limitations.
          </p>
        </div>
      </div>
    </>
  );
}
