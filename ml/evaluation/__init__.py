"""Evaluation package exports."""

from .metrics import AgeMetrics, GenderMetrics, EvaluationResult, compute_age_metrics, compute_gender_metrics, compute_composite
from .evaluator import evaluate_model, make_val_split, ValDataset

__all__ = [
    "AgeMetrics",
    "GenderMetrics",
    "EvaluationResult",
    "compute_age_metrics",
    "compute_gender_metrics",
    "compute_composite",
    "evaluate_model",
    "make_val_split",
    "ValDataset",
]
