"""Face detection utilities."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class FaceDetectorBackend(enum.Enum):
    MEDIA_PIPE = "mediapipe"
    HAAR = "haar"
    NONE = "none"  # fallback: treat full image as face


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.w, self.h

    def expand(self, factor: float) -> "FaceBox":
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        nw = self.w * factor
        nh = self.h * factor
        x1 = int(cx - nw / 2.0)
        y1 = int(cy - nh / 2.0)
        x2 = int(cx + nw / 2.0)
        y2 = int(cy + nh / 2.0)
        return FaceBox(x1, y1, x2, y2, self.confidence)

    def clamp(self, width: int, height: int) -> "FaceBox":
        x1 = max(0, min(width, self.x1))
        y1 = max(0, min(height, self.y1))
        x2 = max(0, min(width, self.x2))
        y2 = max(0, min(height, self.y2))
        return FaceBox(x1, y1, x2, y2, self.confidence)


class FaceDetector:
    def __init__(self, backend: FaceDetectorBackend | str = FaceDetectorBackend.MEDIA_PIPE, min_confidence: float = 0.5) -> None:
        if isinstance(backend, str):
            backend = FaceDetectorBackend(backend)
        self.backend = backend
        self.min_confidence = min_confidence
        self._mp = None
        if self.backend == FaceDetectorBackend.MEDIA_PIPE:
            try:
                import mediapipe as mp  # type: ignore

                self._mp = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence)
            except Exception:
                # fallback
                self.backend = FaceDetectorBackend.HAAR

        self._haar = None
        if self.backend == FaceDetectorBackend.HAAR:
            try:
                self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            except Exception:
                self.backend = FaceDetectorBackend.NONE

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """Detect faces in BGR/uint8 or RGB image."""
        if image is None:
            return []
        # ensure BGR for cv2
        if image.ndim == 3 and image.shape[-1] == 3:
            # if RGB, convert
            if self._is_rgb(image):
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr = image
        elif image.ndim == 3 and image.shape[-1] == 4:
            if self._is_rgb(image):
                bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                bgr = image[:, :, :3]
        else:
            return []
        h, w = bgr.shape[:2]
        boxes: List[FaceBox] = []

        if self.backend == FaceDetectorBackend.MEDIA_PIPE and self._mp is not None:
            results = self._mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if results.detections:
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    conf = det.score[0] if det.score else 0.0
                    if conf < self.min_confidence:
                        continue
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    boxes.append(FaceBox(max(0,x1), max(0,y1), min(w,x2), min(h,y2), float(conf)))
            return self._dedupe(boxes, w, h)

        if self.backend == FaceDetectorBackend.HAAR and self._haar is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            for (x,y,fw,fh) in faces:
                boxes.append(FaceBox(x,y,x+fw,y+fh,1.0))
            return self._dedupe(boxes, w, h)

        # fallback: full image
        boxes.append(FaceBox(0,0,w,h,1.0))
        return boxes

    @staticmethod
    def _is_rgb(img: np.ndarray) -> bool:
        return img.shape[-1] == 3  # heuristic; caller may pass RGB or BGR

    @staticmethod
    def _dedupe(boxes: List[FaceBox], w: int, h: int, iou_thresh: float = 0.5) -> List[FaceBox]:
        if len(boxes) <= 1:
            return boxes
        # simple NMS-like dedupe
        def iou(a: FaceBox, b: FaceBox) -> float:
            ix = max(a.x1,b.x1); iy = max(a.y1,b.y1)
            ex = min(a.x2,b.x2); ey = min(a.y2,b.y2)
            if ex <= ix or ey <= iy: return 0.0
            inter = (ex-ix)*(ey-iy)
            un = a.area + b.area - inter
            return inter/un if un>0 else 0.0
        keep = list(range(len(boxes)))
        # naive: prefer larger area
        order = sorted(range(len(boxes)), key=lambda k: boxes[k].area, reverse=True)
        out = []
        for i in order:
            add = True
            for j in out:
                if iou(boxes[i], boxes[j]) > iou_thresh:
                    add = False; break
            if add: out.append(boxes[i])
        return out
