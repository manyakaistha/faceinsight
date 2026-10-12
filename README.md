# FaceInsight — Vision Research Studio

A local facial age and dataset-label estimation workbench powered by the NPPE-1 ResNet50 + LoRA checkpoint. The interface takes inspiration from Touchline's editorial research desk, with deep teal, ivory, copper, and animated feature diagrams.

## Run locally

Install the Python dependencies into `.venv` and frontend dependencies:

```bash
uv sync --frozen
cd frontend && npm ci
```

Start the backend and web UI in two terminals from the repository root:

```bash
make backend   # http://127.0.0.1:8100, API docs at /docs
make frontend  # http://localhost:3100
```

The Next.js server proxies `/api/v1/*` to the backend. Set `FACEINSIGHT_API_URL` in the frontend environment to override `http://127.0.0.1:8100`. Ports 3100 / 8100 keep this project separate from applications using 3000 / 8000.

For a production frontend:

```bash
cd frontend
npm run build
npm start
```

## Workspace

- **Overview:** entry points, model facts, and a responsive animated feature diagram.
- **Live inference:** JPG, PNG, and WebP uploads (up to 10 MB); multiple detected faces, estimates, and JSON download.
- **Explainability:** live age or dataset-label Grad-CAM; source, heatmap, overlay, and PNG download.
- **Performance:** saved evaluation selector, age-error charts, classification matrix, and report download. Archived reports predate the corrected output-layer loader and are identified as historical.
- **Architecture:** interactive pipeline and live parameter counts.
- **About:** training context, interpretation, and limitations.

Canvas motion uses seeded geometry, respects reduced-motion settings, and pauses when the canvas is offscreen or the page is hidden. Uploaded portraits are processed in memory and are not saved by the backend.

## Model files

Model checkpoints, training data, and original notebooks are excluded from this source repository. To run inference, put the NPPE-1 champion `weights_state_dict.pth` and `model_meta.json` in `export_resnet18_lora_v6/`. The folder's legacy name says ResNet18; its metadata and checkpoint identify ResNet50. The model loader requires every checkpoint key to match. The three optional sample portraits belong in `face_dataset/test/00000.jpg` through `00002.jpg`.

The heads preserve both dropout layers from the training notebook. See [`docs/model_card.md`](docs/model_card.md) for intended use and model limitations.

## Development checks

```bash
.venv/bin/python -m pytest -q
cd frontend && npx tsc --noEmit
```

Predicted Female / Male labels follow the dataset taxonomy; they do not determine gender identity. This model has no demographic fairness validation and is unsuitable for high-stakes decisions.
