import Link from "next/link";
import {
  ArrowUpRight,
  ScanLine,
  Fingerprint,
  ChartNoAxesCombined,
  Network,
  ArrowRight,
} from "lucide-react";
import Field from "../components/field";
const cards = [
  {
    href: "/inference",
    icon: ScanLine,
    n: "01",
    title: "Look a little closer",
    text: "Upload a portrait. Explore age estimates and predicted dataset labels.",
  },
  {
    href: "/explainability",
    icon: Fingerprint,
    n: "02",
    title: "See what the model sees",
    text: "Generate a Grad-CAM map and inspect the regions behind an output.",
  },
  {
    href: "/performance",
    icon: ChartNoAxesCombined,
    n: "03",
    title: "Let the evidence speak",
    text: "Browse saved evaluation runs, errors, and classification outcomes.",
  },
  {
    href: "/architecture",
    icon: Network,
    n: "04",
    title: "Follow the connections",
    text: "Explore the ResNet50 backbone, low-rank adapters, and dual heads.",
  },
];
export default function Home() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <span /> THE HUMAN FACE, THROUGH A MODEL
          </div>
          <h1>
            Every face.
            <br />A new <em>perspective.</em>
          </h1>
          <p>
            A thoughtful space to explore computer vision. Look beyond the
            prediction, follow the features, and understand the model behind
            them.
          </p>
          <div className="hero-actions">
            <Link href="/inference" className="button primary">
              Explore a portrait <ArrowUpRight size={17} />
            </Link>
            <Link href="/architecture" className="text-link">
              Meet the model <ArrowRight size={15} />
            </Link>
          </div>
          <div className="capabilities">
            <span>
              <i />
              Real checkpoint inference
            </span>
            <span>
              <i />
              Visual explanations
            </span>
            <span>
              <i />
              Runs locally
            </span>
          </div>
        </div>
        <div className="hero-art">
          <span className="art-label">A STUDY IN FEATURES</span>
          <Field />
          <div className="art-foot">
            <span className="tiny-orbit" /> ResNet50 + low-rank adaptation{" "}
            <span>↗</span>
          </div>
        </div>
      </section>
      <section className="stat-strip">
        <div>
          <span>BACKBONE</span>
          <strong>
            ResNet50 <small>50-layer residual network</small>
          </strong>
        </div>
        <div>
          <span>INPUT RESOLUTION</span>
          <strong>
            224 × 224 <small>Normalized RGB face crop</small>
          </strong>
        </div>
        <div>
          <span>ADAPTER RANK</span>
          <strong>
            12 <small>α = 24 · scaling = 2</small>
          </strong>
        </div>
        <div>
          <span>OUTPUTS</span>
          <strong>
            Two perspectives <small>Age + binary dataset label</small>
          </strong>
        </div>
      </section>
      <div className="section-heading">
        <div className="eyebrow">YOUR RESEARCH DESK</div>
        <span>Choose a way in ↘</span>
      </div>
      <section className="explore-grid">
        {cards.map((c) => (
          <Link href={c.href} className="explore-card" key={c.n}>
            <div className="card-top">
              <c.icon size={22} />
              <span>{c.n}</span>
            </div>
            <h2>{c.title}</h2>
            <p>{c.text}</p>
            <span className="card-arrow">
              <ArrowUpRight size={19} />
            </span>
          </Link>
        ))}
      </section>
      <div className="home-note">
        <span className="note-mark">✳</span>
        <p>
          <strong>Curiosity, with context.</strong> Predictions reflect patterns
          learned from training data. Explore them alongside model explanations
          and evaluation evidence.
        </p>
        <Link href="/about">
          Read the model notes <ArrowUpRight size={14} />
        </Link>
      </div>
    </>
  );
}
