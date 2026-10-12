import Link from "next/link";
import { ArrowUpRight, ScanFace, Fingerprint, Laptop } from "lucide-react";
import { PageHeading } from "../../components/workbench";
export default function Page() {
  return (
    <>
      <PageHeading
        eyebrow="CURIOSITY, WITH CONTEXT"
        title={
          <>
            A closer look at <em>vision.</em>
          </>
        }
        description="FaceInsight turns a deep learning experiment into a transparent, hands-on research workspace."
      />
      <div className="about-intro">
        <span className="about-symbol">
          <ScanFace size={64} strokeWidth={1} />
        </span>
        <div>
          <h2>
            Understand the model.
            <br />
            Question the estimate.
          </h2>
          <p>
            Built from the NPPE-1 facial age and gender estimation project, this
            studio connects a ResNet50 + LoRA checkpoint to live predictions,
            visual explanations, and recorded evaluation evidence.
          </p>
        </div>
      </div>
      <div className="two-column">
        <section className="panel prose-panel">
          <div className="eyebrow">THE EXPERIMENT</div>
          <h2>One backbone. Two tasks.</h2>
          <p>
            The model learns shared image features and produces an age estimate
            and a binary label defined by the training dataset. Training used
            approximately 34,708 labeled images, with an imbalanced label
            distribution of about 75.2% Male and 24.8% Female.
          </p>
          <p>
            The exported checkpoint comes from epoch 22. The legacy export
            folder name mentions ResNet18; its metadata and weights identify the
            actual backbone as ResNet50.
          </p>
          <Link href="/architecture" className="text-link">
            Explore the architecture <ArrowUpRight size={15} />
          </Link>
        </section>
        <section className="panel prose-panel">
          <div className="eyebrow">READING THE OUTPUTS</div>
          <h2>Estimates need context.</h2>
          <p>
            Age is a visual estimate in the 0–100 range. The classification
            follows the dataset’s Female / Male labels; it does not determine
            gender identity or biological sex.
          </p>
          <p>
            Lighting, pose, occlusion, image quality, and training-data
            imbalance can affect results. Output probabilities are not
            calibrated confidence scores. Grad-CAM highlights relative model
            activation, rather than proving why a prediction is correct.
          </p>
          <Link href="/performance" className="text-link">
            Review the evidence <ArrowUpRight size={15} />
          </Link>
        </section>
      </div>
      <section className="principles">
        <div>
          <Laptop size={21} />
          <h3>A local workspace</h3>
          <p>
            Images are sent to your local model server. The inference endpoints
            process them in memory and do not save uploaded portraits.
          </p>
        </div>
        <div>
          <Fingerprint size={21} />
          <h3>Show the attention</h3>
          <p>
            Live explanations expose the final convolutional layer’s activation
            for the selected prediction task.
          </p>
        </div>
        <div>
          <ScanFace size={21} />
          <h3>Research, thoughtfully</h3>
          <p>
            Use this project for learning and exploration. It has no demographic
            fairness validation and is unsuitable for high-stakes decisions.
          </p>
        </div>
      </section>
      <section className="panel about-links">
        <div>
          <div className="eyebrow">CONTINUE EXPLORING</div>
          <h2>Your next perspective awaits.</h2>
        </div>
        <Link href="/inference" className="button primary">
          Start an analysis <ArrowUpRight size={16} />
        </Link>
      </section>
      <p className="credit">
        Interface direction inspired by{" "}
        <a
          href="https://github.com/manyakaistha/touchline-fifa-worldcup-hybrid-graph-rag"
          target="_blank"
          rel="noreferrer"
        >
          Touchline’s research workbench ↗
        </a>
        , adapted for a computer vision studio.
      </p>
    </>
  );
}
