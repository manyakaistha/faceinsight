"""Evaluation runner for held-out validation set."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..inference import FaceInsightInferencer
from ..models import ModelConfig
from .metrics import EvaluationResult, compute_age_metrics, compute_gender_metrics, compute_composite


class ValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str | Path, img_dir: str = "train", img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.root = Path(root_dir)
        self.img_dir = img_dir
        self.img_size = img_size
        from torchvision import transforms

        self.tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = str(r["id"]).zfill(5)
        p = self.root / self.img_dir / f"{pid}.jpg"
        if not p.exists():
            # try other
            for ext in ["jpeg", "png"]:
                pp = self.root / self.img_dir / f"{pid}.{ext}"
                if pp.exists():
                    p = pp
                    break
        img = Image.open(p).convert("RGB")
        t = self.tfm(img)
        return t, float(r["age"]), int(r["gender"]), pid


def make_val_split(csv_path: str | Path, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"id": str})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = int(n * val_frac)
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


def evaluate_model(
    inferencer: Optional[FaceInsightInferencer] = None,
    val_df: Optional[pd.DataFrame] = None,
    root_dir: str | Path = "face_dataset",
    img_dir: str = "train",
    batch_size: int = 32,
    device: str = "cpu",
) -> EvaluationResult:
    if inferencer is None:
        inferencer = FaceInsightInferencer(device=device)
    if val_df is None:
        train_csv = Path(root_dir) / "train.csv"
        _, val_df = make_val_split(train_csv)
    ds = ValDataset(val_df, root_dir=root_dir, img_dir=img_dir, img_size=inferencer.config.img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_age_t, y_age_p, y_g_t, y_g_p = [], [], [], []
    for t, age_t, g_t, pid in dl:
        t = t.to(inferencer.device)
        age_tn, prob_tn = inferencer._predict_tensor(t)  # type: ignore[attr-defined]
        age_pv = torch.clamp(age_tn, inferencer.config.min_age, inferencer.config.max_age)
        y_age_t.extend(age_t.numpy())
        y_age_p.extend(age_pv.cpu().numpy())
        g_pv = (prob_tn >= inferencer.decision_threshold).long()
        y_g_t.extend(g_t.numpy())
        y_g_p.extend(g_pv.cpu().numpy())
    am = compute_age_metrics(np.array(y_age_t), np.array(y_age_p))
    gm = compute_gender_metrics(np.array(y_g_t), np.array(y_g_p))
    aq, cs = compute_composite(gm.macro_f1, am.rmse, inferencer.config.age_span)
    return EvaluationResult(
        num_samples=len(val_df),
        age=am,
        gender=gm,
        age_quality=aq,
        composite_score=cs,
        model_version=inferencer.config.version,
    )
