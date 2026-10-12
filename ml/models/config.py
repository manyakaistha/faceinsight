"""Model configuration objects and architecture registry."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ArchitectureSpec:
    """Static description of a supported backbone."""

    name: str
    out_dim: int
    family: str  # "resnet" | "efficientnet"


ARCHITECTURES: Dict[str, ArchitectureSpec] = {
    "resnet18": ArchitectureSpec("resnet18", 512, "resnet"),
    "resnet34": ArchitectureSpec("resnet34", 512, "resnet"),
    "resnet50": ArchitectureSpec("resnet50", 2048, "resnet"),
    "efficientnet_b0": ArchitectureSpec("efficientnet_b0", 1280, "efficientnet"),
}


@dataclass
class ModelConfig:
    """Configuration required to instantiate and run the multi-task model.

    Defaults match the champion ``export_resnet18_lora_v6`` checkpoint.
    """

    arch: str = "resnet50"
    img_size: int = 224
    mean: List[float] = field(default_factory=lambda: list(IMAGENET_MEAN))
    std: List[float] = field(default_factory=lambda: list(IMAGENET_STD))
    min_age: float = 0.0
    max_age: float = 100.0
    lora_r: int = 12
    lora_alpha: float = 24.0
    lora_dropout: float = 0.0
    head_dropout: float = 0.2
    age_weight: float = 0.7
    gender_weight: float = 0.3
    gender_threshold: float = 0.5
    version: str = "champion-resnet50-lora-v6"
    checkpoint: Optional[str] = None

    @property
    def family(self) -> str:
        return ARCHITECTURES[self.arch].family

    @property
    def out_dim(self) -> int:
        return ARCHITECTURES[self.arch].out_dim

    @property
    def age_span(self) -> float:
        return self.max_age - self.min_age

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    @classmethod
    def from_meta(cls, meta: dict, **overrides) -> "ModelConfig":
        """Build from a ``model_meta.json`` produced by the training notebooks."""
        cfg = cls(
            arch=meta.get("arch", "resnet50"),
            img_size=meta.get("img_size", 224),
            mean=meta.get("mean", list(IMAGENET_MEAN)),
            std=meta.get("std", list(IMAGENET_STD)),
            min_age=meta.get("min_age", 0.0),
            max_age=meta.get("max_age", 100.0),
            lora_r=meta.get("lora_r", 12),
            lora_alpha=meta.get("lora_alpha", 24.0),
            age_weight=meta.get("age_weight", 0.7),
            gender_weight=meta.get("gender_weight", 0.3),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
