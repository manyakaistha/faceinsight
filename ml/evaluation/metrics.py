"""Evaluation metrics and reporting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
)


@dataclass
class AgeMetrics:
    mae: float
    rmse: float
    median_ae: float
    max_error: float
    min_error: float
    std_error: float


@dataclass
class GenderMetrics:
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    confusion_matrix: List[List[int]]


@dataclass
class EvaluationResult:
    num_samples: int
    age: AgeMetrics
    gender: GenderMetrics
    age_quality: float
    composite_score: float
    model_version: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "age": asdict(self.age),
            "gender": {
                **asdict(self.gender),
                "confusion_matrix": self.gender.confusion_matrix,
            },
            "age_quality": self.age_quality,
            "composite_score": self.composite_score,
            "model_version": self.model_version,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_age_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> AgeMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    return AgeMetrics(
        mae=float(mae),
        rmse=float(rmse),
        median_ae=float(medae),
        max_error=float(np.max(errors)),
        min_error=float(np.min(errors)),
        std_error=float(np.std(errors)),
    )


def compute_gender_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> GenderMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return GenderMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=cm,
    )


def compute_composite(gender_macro_f1: float, rmse: float, age_span: float = 100.0) -> Tuple[float, float]:
    age_span = max(age_span, 1e-6)
    age_quality = max(1.0 - rmse / age_span, 1e-6)
    denom = gender_macro_f1 + age_quality + 1e-12
    score = (2.0 * gender_macro_f1 * age_quality) / denom
    return float(age_quality), float(score)
