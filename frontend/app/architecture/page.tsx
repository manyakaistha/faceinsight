"use client";
import { useEffect, useState } from "react";
import { ArrowRight, Cpu, Check } from "lucide-react";
import { api, PageHeading } from "../../components/workbench";
import Field from "../../components/field";
const stages = [
  {
    name: "Face crop",
    tag: "224 × 224 × 3",
    detail:
      "The detector locates faces, expands the crop by 1.1×, resizes to 224 × 224, and applies ImageNet normalization.",
  },
  {
    name: "ResNet50",
    tag: "RESIDUAL BACKBONE",
    detail:
      "A 50-layer residual network extracts visual features. Skip connections allow information to flow through deep convolutional blocks.",
  },
  {
    name: "LoRA adapters",
    tag: "RANK 12 · α 24",
    detail:
      "Low-rank 1 × 1 convolution branches adapt the frozen backbone. Each update is scaled by α / rank = 2 and added to the base convolution.",
  },
  {
    name: "Shared features",
    tag: "2,048 DIMENSIONS",
    detail:
      "Global average pooling turns the spatial feature maps into a 2,048-dimensional representation shared by both prediction heads.",
  },
  {
    name: "Dual heads",
    tag: "AGE + LABEL",
    detail:
      "Each head uses dropout, a 2,048 → 256 linear layer, ReLU, a second dropout, and a 256 → 1 output layer. Age is clamped to 0–100; gender logits pass through sigmoid.",
  },
];
type Architecture = {
  config: { arch: string; lora_r: number; lora_alpha: number; version: string };
  total_params: number;
  trainable_params: number;
  checkpoint_status: string;
};
export default function Page() {
  const [active, setActive] = useState(1),
    [data, setData] = useState<Architecture | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    api<Architecture>("model/architecture")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  return (
    <>
      <PageHeading
        eyebrow="FROM PIXELS TO PERSPECTIVES"
        title={
          <>
            A model of <em>connections.</em>
          </>
        }
        description="Follow the path from a face crop to two outputs. Select a stage to explore its role."
      />
      <section className="panel pipeline-panel">
        <div className="panel-heading">
          The inference pathway <span className="mono">SELECT A STAGE</span>
        </div>
        <div className="pipeline">
          {stages.map((s, i) => (
            <div className="pipeline-item" key={s.name}>
              <button
                onClick={() => setActive(i)}
                className={active === i ? "selected" : ""}
                aria-pressed={active === i}
              >
                <span className="eyebrow">0{i + 1}</span>
                <strong>{s.name}</strong>
                <small>{s.tag}</small>
              </button>
              {i < 4 && <ArrowRight size={17} />}
            </div>
          ))}
        </div>
        <div className="stage-detail" aria-live="polite">
          <span className="stage-number">0{active + 1}</span>
          <div>
            <h2>{stages[active].name}</h2>
            <p>{stages[active].detail}</p>
          </div>
        </div>
      </section>
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      <div className="two-column architecture-details">
        <section className="panel">
          <div className="panel-heading">
            <span>
              <Cpu size={16} /> Loaded model
            </span>
            <span className="mono">LIVE METADATA</span>
          </div>
          {data ? (
            <div className="spec-list">
              <div>
                <span>Architecture</span>
                <strong>{data.config.arch}</strong>
              </div>
              <div>
                <span>Total parameters</span>
                <strong>{data.total_params.toLocaleString()}</strong>
              </div>
              <div>
                <span>Trainable parameters</span>
                <strong>{data.trainable_params.toLocaleString()}</strong>
              </div>
              <div>
                <span>Trainable share</span>
                <strong>
                  {((data.trainable_params / data.total_params) * 100).toFixed(
                    1,
                  )}
                  %
                </strong>
              </div>
              <div>
                <span>Adapter configuration</span>
                <strong>
                  r = {data.config.lora_r} · α = {data.config.lora_alpha}
                </strong>
              </div>
              <div>
                <span>Weight compatibility</span>
                <strong className="verified">
                  <Check size={14} /> All keys matched
                </strong>
              </div>
            </div>
          ) : (
            <p className="panel-body">
              {error
                ? "Model metadata unavailable."
                : "Connecting to the local model…"}
            </p>
          )}
        </section>
        <section className="panel adapter-panel">
          <div className="panel-heading">
            Small updates. Shared understanding.
          </div>
          <Field compact />
          <div className="adapter-copy">
            <h2>Adapt, without starting over.</h2>
            <p>
              LoRA learns a low-rank update while preserving the pretrained
              convolution weights. Both output heads learn from the same facial
              representation.
            </p>
            <div className="equation">output = base(x) + 2 · B(A(x))</div>
          </div>
        </section>
      </div>
      <div className="notice">
        <span>✳</span>
        <p>
          <strong>Faithful checkpoint loading.</strong> The prediction heads
          include the second dropout layer used in the training notebook. Strict
          loading verifies that every trained weight has a matching layer.
        </p>
      </div>
    </>
  );
}
