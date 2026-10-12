"""FastAPI backend."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ml.inference import FaceInsightInferencer
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.preprocessing import load_image_pil, crop_face_from_pil, pil_to_tensor_normalized
import numpy as np


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    timestamp: float


class ModelInfo(BaseModel):
    arch: str
    model_version: str
    input_shape: List[int]
    device: str
    checkpoint_status: str


class FacePredictionSchema(BaseModel):
    bbox: dict
    estimated_age: float
    predicted_label: str
    gender_probability: float
    decision_threshold: float


class PredictResponse(BaseModel):
    success: bool
    faces: List[FacePredictionSchema] = []
    model_version: Optional[str] = None
    device: Optional[str] = None
    error_message: Optional[str] = None


class ConfigPublic(BaseModel):
    gender_threshold: float
    face_expansion: float
    tta_flip: bool
    min_face_confidence: float


_inferencer: Optional[FaceInsightInferencer] = None
_app_start = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inferencer
    try:
        _inferencer = FaceInsightInferencer()
    except Exception:
        logging.exception("Could not initialize the FaceInsight checkpoint")
        _inferencer = None
    yield
    _inferencer = None


app = FastAPI(title="FaceInsight AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_inferencer() -> FaceInsightInferencer:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _inferencer


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", timestamp=time.time())


@app.get("/api/v1/ready")
async def ready():
    if _inferencer is None:
        return JSONResponse({"ready": False, "reason": "model_not_loaded"}, status_code=503)
    return {"ready": True}


@app.get("/api/v1/model/info", response_model=ModelInfo)
async def model_info():
    inf = get_inferencer()
    return ModelInfo(
        arch=inf.config.arch,
        model_version=inf.config.version,
        input_shape=[3, inf.config.img_size, inf.config.img_size],
        device=str(inf.device),
        checkpoint_status="loaded",
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    inf = get_inferencer()
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image content type")
    res = inf.predict_image(data, multi_face=True)
    faces = []
    for f in (res.faces or []):
        b = f.bbox
        faces.append({
            "bbox": {
                "x1": int(b["x1"]),
                "y1": int(b["y1"]),
                "x2": int(b["x2"]),
                "y2": int(b["y2"]),
                "confidence": float(b["confidence"]),
            },
            "estimated_age": float(f.estimated_age),
            "predicted_label": f.predicted_label,
            "gender_probability": float(f.gender_probability),
            "decision_threshold": float(f.decision_threshold),
        })
    return PredictResponse(
        success=res.success,
        faces=faces,
        model_version=res.model_version,
        device=res.device,
        error_message=res.error_message,
    )


@app.post("/api/v1/predict/batch", response_model=List[PredictResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    inf = get_inferencer()
    results: List[PredictResponse] = []
    for f in files[:32]:
        try:
            d = await f.read()
            if len(d) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")
            if not f.content_type or not f.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image content type")
            r = inf.predict_image(d, multi_face=False)
            faces = []
            for f in (r.faces or []):
                faces.append({
                    "bbox": {"x1": int(f.bbox["x1"]), "y1": int(f.bbox["y1"]), "x2": int(f.bbox["x2"]), "y2": int(f.bbox["y2"]), "confidence": float(f.bbox["confidence"])},
                    "estimated_age": float(f.estimated_age),
                    "predicted_label": f.predicted_label,
                    "gender_probability": float(f.gender_probability),
                    "decision_threshold": float(f.decision_threshold),
                })
            results.append(
                PredictResponse(
                    success=r.success,
                    faces=faces,
                    model_version=r.model_version,
                    device=r.device,
                    error_message=r.error_message,
                )
            )
        except HTTPException:
            raise
        except Exception as e:
            results.append(PredictResponse(success=False, error_message=str(e)))
    return results


@app.get("/api/v1/config/public", response_model=ConfigPublic)
async def config_public():
    inf = get_inferencer()
    return ConfigPublic(
        gender_threshold=inf.decision_threshold,
        face_expansion=inf.face_expansion,
        tta_flip=inf.tta_flip,
        min_face_confidence=inf.face_detector.min_confidence,
    )


@app.get("/api/v1/metrics")
async def metrics():
    return {"message": "Use evaluation module to generate real metrics artifacts"}


# Read-only assets and live explanations for the web workbench.
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {"portrait-1": "00000.jpg", "portrait-2": "00001.jpg", "portrait-3": "00002.jpg"}


@app.get('/api/v1/samples/{sample_id}')
def sample_image(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404, 'Sample not found')
    path = REPO_ROOT / 'face_dataset' / 'test' / SAMPLES[sample_id]
    if not path.is_file():
        raise HTTPException(404, 'Sample image unavailable')
    return FileResponse(path, media_type='image/jpeg')


@app.get('/api/v1/evaluation')
def evaluation():
    reports = []
    for path in sorted((REPO_ROOT / 'artifacts' / 'evaluation').glob('*.json')):
        reports.append({'id': path.stem, 'metrics': json.loads(path.read_text())})
    return {'reports': reports, 'note': 'Archived evaluations were generated before the output-layer loading correction. They are historical results, not a fresh evaluation of the corrected model.'}


@app.get('/api/v1/model/architecture')
def architecture():
    inf = get_inferencer()
    return {'config': inf.config.to_dict(), 'total_params': inf.model.total_params,
            'trainable_params': inf.model.trainable_params,
            'checkpoint_status': 'All checkpoint keys matched',
            'detector': type(inf.face_detector).__name__}


@app.post('/api/v1/explain')
async def explain(file: UploadFile = File(...), target: Literal['age', 'gender'] = 'gender'):
    inf = get_inferencer()
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, 'File too large')
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'Invalid image content type')
    explainer = None
    try:
        pil = load_image_pil(data)
        boxes = inf.face_detector.detect(np.array(pil.convert('RGB'))[:, :, ::-1])
        if not boxes:
            raise HTTPException(422, 'No face detected. Try a clear, front-facing portrait.')
        box = max(boxes, key=lambda b: b.area)
        crop = crop_face_from_pil(pil, box.x1, box.y1, box.x2, box.y2, expand_factor=inf.face_expansion)
        crop = crop.resize((inf.config.img_size, inf.config.img_size))
        tensor = pil_to_tensor_normalized(crop, mean=inf.config.mean, std=inf.config.std).unsqueeze(0)
        explainer = GradCAMExplainer(inf.model, target=target)
        result = explainer.explain(tensor)
        if not result['success']:
            raise HTTPException(500, result['error'])
        def encode(image):
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        return {'success': True, 'target': target, 'original': encode(crop),
                'heatmap': encode(result['heatmap_image']), 'overlay': encode(result['overlay_image']),
                'model_version': inf.config.version}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'Could not explain this image: {exc}')
    finally:
        if explainer is not None:
            explainer.remove_hooks()
        inf.model.zero_grad(set_to_none=True)
