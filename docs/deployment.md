# Deployment

## Local workbench

From the repository root, run `make backend` and `make frontend` in separate terminals. The UI runs at http://localhost:3100 and FastAPI runs at http://127.0.0.1:8100 (documentation at `/docs`).

For a production frontend, run `npm run build` then `npm start` inside `frontend/`. Keep the backend running separately. The Next.js proxy uses `FACEINSIGHT_API_URL`, defaulting to http://127.0.0.1:8100.

## Docker backend

`docker compose up --build` starts the existing backend container on port 8000. To connect the local frontend to it, run `FACEINSIGHT_API_URL=http://127.0.0.1:8000 npm run dev` inside `frontend/`.

## Model and sample artifacts

Keep `export/export_resnet18_lora_v6/weights_state_dict.pth` and `model_meta.json` in place. Live inference uses the checkpoint without remote services. Sample portraits require `face_dataset/test/00000.jpg`, `00001.jpg`, and `00002.jpg`. Evaluation pages read the JSON reports in `artifacts/evaluation/`.
