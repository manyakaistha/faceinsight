"""Tests for ML models."""

from ml.inference import FaceInsightInferencer


def test_inferencer_loads_champion():
    inf = FaceInsightInferencer()
    assert inf.model is not None
    assert inf.config.arch in ("resnet50", "resnet18")
    assert inf.config.img_size == 224
