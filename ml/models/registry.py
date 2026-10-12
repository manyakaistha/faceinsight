"""Model registry utilities, checkpoint loading and export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .config import ModelConfig, ARCHITECTURES, IMAGENET_MEAN, IMAGENET_STD
from .multitask import AgeGenderModel


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> Dict:
    ck = torch.load(str(checkpoint_path), map_location=map_location, weights_only=weights_only)
    return ck


def build_model_from_config(cfg: ModelConfig, pretrained_backbone: bool = False) -> AgeGenderModel:
    return AgeGenderModel(
        arch=cfg.arch,
        pretrained=pretrained_backbone,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        head_dropout=cfg.head_dropout,
        inject_first=True,
    )


def load_model_from_checkpoint_dir(
    checkpoint_dir: str | Path,
    device: str | torch.device = "cpu",
    prefer_weights_state_dict: bool = True,
) -> Tuple[AgeGenderModel, ModelConfig]:
    cdir = Path(checkpoint_dir)
    meta_path = cdir / "model_meta.json"
    wsd_path = cdir / "weights_state_dict.pth"
    best_path = None
    # find best by score
    for p in cdir.glob("checkpoint_e*_s*.pth"):
        best_path = p if best_path is None else max(best_path, p, key=lambda x: float(x.stem.split("s")[-1]))
    weights_path = wsd_path if wsd_path.exists() and prefer_weights_state_dict else (best_path or wsd_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights found in {cdir}")

    meta = {}
    if meta_path.exists():
        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    cfg = ModelConfig.from_meta(meta)
    model = build_model_from_config(cfg, pretrained_backbone=False)
    sd = load_checkpoint(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def load_champion_model(device: str | torch.device = "cpu") -> Tuple[AgeGenderModel, ModelConfig]:
    # Look in common locations
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "export_resnet18_lora_v6",
        repo_root / "checkpoints" / "champion",
        Path("export_resnet18_lora_v6"),
    ]
    for c in candidates:
        if c.exists():
            return load_model_from_checkpoint_dir(c, device=device)
    raise FileNotFoundError("Champion checkpoint directory not found (expected export_resnet18_lora_v6/)")
