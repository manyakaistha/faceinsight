"""Inference engine: single/multi-face prediction with optional TTA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..models import ModelConfig, load_champion_model
from .face_detection import FaceBox, FaceDetector, FaceDetectorBackend
from .preprocessing import crop_face_from_pil, load_image_pil, pil_to_tensor_normalized


@dataclass
class FacePrediction:
    bbox: Dict[str, float]
    estimated_age: float
    predicted_label: str  # "Female" or "Male"
    gender_probability: float
    decision_threshold: float


@dataclass
class InferenceResult:
    success: bool
    error_message: Optional[str] = None
    faces: List[FacePrediction] = None
    model_version: Optional[str] = None
    device: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "error_message": self.error_message,
            "faces": [asdict(f) for f in (self.faces or [])],
            "model_version": self.model_version,
            "device": self.device,
        }
        return d


class FaceInsightInferencer:
    def __init__(
        self,
        model=None,
        config: Optional[ModelConfig] = None,
        device: str | torch.device | None = None,
        face_detector_backend: Union[str, FaceDetectorBackend] = FaceDetectorBackend.MEDIA_PIPE,
        face_expansion: float = 1.1,
        tta_flip: bool = False,
        min_face_confidence: float = 0.5,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        if model is None or config is None:
            m, c = load_champion_model(self.device)
            self.model = m
            self.config = c
        else:
            self.model = model.to(self.device)
            self.config = config if config is not None else ModelConfig()
        self.model.eval()
        self.face_detector = FaceDetector(backend=face_detector_backend, min_confidence=min_face_confidence)
        self.face_expansion = face_expansion
        self.tta_flip = tta_flip
        self._decision_threshold = self.config.gender_threshold

    @property
    def decision_threshold(self) -> float:
        return self._decision_threshold

    @decision_threshold.setter
    def decision_threshold(self, v: float) -> None:
        self._decision_threshold = max(0.01, min(0.99, float(v)))

    @torch.inference_mode()
    def _predict_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tta_flip:
            # TTA: original + horizontal flip, average
            age0, p0 = self.model.predict(x)
            xf = torch.flip(x, dims=[3])
            age1, p1 = self.model.predict(xf)
            age = (age0 + age1) / 2.0
            p = (p0 + p1) / 2.0
            return age, p
        return self.model.predict(x)

    def predict_image(self, data: Union[str, Path, bytes, Image.Image], multi_face: bool = True) -> InferenceResult:
        try:
            pil = load_image_pil(data)
            arr_bgr = np.array(pil.convert("RGB"))
            arr_bgr = arr_bgr[:, :, ::-1]  # RGB->BGR for detector
            boxes = self.face_detector.detect(arr_bgr)
            if not boxes:
                return InferenceResult(success=False, error_message="No face detected in the image")
            if not multi_face and len(boxes) > 1:
                # take largest
                boxes = [max(boxes, key=lambda b: b.area)]
            faces_out: List[FacePrediction] = []
            for b in boxes:
                face_pil = crop_face_from_pil(pil, b.x1, b.y1, b.x2, b.y2, expand_factor=self.face_expansion)
                face_pil = face_pil.resize((self.config.img_size, self.config.img_size))
                t = pil_to_tensor_normalized(face_pil, mean=self.config.mean, std=self.config.std)
                t = t.unsqueeze(0).to(self.device, dtype=torch.float32)
                age_t, prob_t = self._predict_tensor(t)
                age_v = float(age_t.squeeze().item())
                prob_v = float(prob_t.squeeze().item())
                if age_v < self.config.min_age:
                    age_v = self.config.min_age
                if age_v > self.config.max_age:
                    age_v = self.config.max_age
                label = "Male" if prob_v >= self.decision_threshold else "Female"
                faces_out.append(
                    FacePrediction(
                        bbox={"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "confidence": float(b.confidence)},
                        estimated_age=float(f"{age_v:.2f}"),
                        predicted_label=label,
                        gender_probability=float(f"{prob_v:.4f}"),
                        decision_threshold=self.decision_threshold,
                    )
                )
            return InferenceResult(success=True, faces=faces_out, model_version=self.config.version, device=str(self.device))
        except Exception as e:
            return InferenceResult(success=False, error_message=str(e))

    def predict_single(self, data: Union[str, Path, bytes, Image.Image]) -> InferenceResult:
        return self.predict_image(data, multi_face=False)

    def predict_batch(self, items: List[Union[str, Path, bytes, Image.Image]], multi_face: bool = False) -> List[InferenceResult]:
        out: List[InferenceResult] = []
        for it in items:
            try:
                out.append(self.predict_image(it, multi_face=multi_face))
            except Exception as e:
                out.append(InferenceResult(success=False, error_message=str(e)))
        return out
