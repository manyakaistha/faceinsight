"""Multi-task model definition for NPPE-1."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from .lora import LoRAConv2d
from .config import ARCHITECTURES


class AgeGenderModel(nn.Module):
    """ResNet-based backbone + LoRA + dual heads (age regression, gender logits).

    Heads follow the recovered specification:
    - Shared features: ``out_dim`` (ResNet50 -> 2048)
    - Age head: ``Dropout(head_dropout) -> Linear(out_dim, 256) -> ReLU -> Dropout(head_dropout / 2) -> Linear(256, 1)``
    - Gender head: ``Dropout(head_dropout) -> Linear(out_dim, 256) -> ReLU -> Dropout(head_dropout / 2) -> Linear(256, 1)``
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        lora_r: int = 12,
        lora_alpha: float = 24.0,
        lora_dropout: float = 0.0,
        head_dropout: float = 0.2,
        inject_first: bool = True,
    ) -> None:
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture {arch}; valid: {list(ARCHITECTURES)}")
        spec = ARCHITECTURES[arch]
        self.arch = arch
        self.spec = spec
        self.out_dim = spec.out_dim

        # Load backbone
        weights = None
        if pretrained:
            if arch == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif arch == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif arch == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            elif arch == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        if spec.family == "resnet":
            if arch == "resnet18":
                self.backbone = models.resnet18(weights=weights)
            elif arch == "resnet34":
                self.backbone = models.resnet34(weights=weights)
            elif arch == "resnet50":
                self.backbone = models.resnet50(weights=weights)
            else:
                raise ValueError(f"Unknown resnet arch {arch}")
            self.backbone.fc = nn.Identity()
        elif spec.family == "efficientnet":
            self.backbone = models.efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown family {spec.family}")

        # Inject LoRA
        self._inject_lora(
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, include_first=inject_first
        )

        # Freeze non-LoRA
        self._freeze_non_lora()

        # Heads
        self.age_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.out_dim, 256),
            nn.ReLU(True),
            nn.Dropout(head_dropout * 0.5),
            nn.Linear(256, 1),
        )
        self.gen_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.out_dim, 256),
            nn.ReLU(True),
            nn.Dropout(head_dropout * 0.5),
            nn.Linear(256, 1),
        )

    def _inject_lora(self, lora_r: int, lora_alpha: float, lora_dropout: float, include_first: bool) -> None:
        if self.spec.family != "resnet":
            return
        if include_first and isinstance(self.backbone.conv1, nn.Conv2d):
            self.backbone.conv1 = LoRAConv2d(self.backbone.conv1, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        from torchvision.models.resnet import BasicBlock, Bottleneck

        for m in self.backbone.modules():
            if isinstance(m, Bottleneck):
                if isinstance(m.conv1, nn.Conv2d):
                    m.conv1 = LoRAConv2d(m.conv1, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                if isinstance(m.conv2, nn.Conv2d):
                    m.conv2 = LoRAConv2d(m.conv2, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                if isinstance(m.conv3, nn.Conv2d):
                    m.conv3 = LoRAConv2d(m.conv3, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                if m.downsample is not None and isinstance(m.downsample, nn.Sequential) and isinstance(m.downsample[0], nn.Conv2d):
                    m.downsample[0] = LoRAConv2d(m.downsample[0], r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            elif isinstance(m, BasicBlock):
                if isinstance(m.conv1, nn.Conv2d):
                    m.conv1 = LoRAConv2d(m.conv1, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                if isinstance(m.conv2, nn.Conv2d):
                    m.conv2 = LoRAConv2d(m.conv2, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                if m.downsample is not None and isinstance(m.downsample, nn.Sequential) and isinstance(m.downsample[0], nn.Conv2d):
                    m.downsample[0] = LoRAConv2d(m.downsample[0], r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def _freeze_non_lora(self) -> None:
        for n, p in self.named_parameters():
            if "lora_A" not in n and "lora_B" not in n and "age_head" not in n and "gen_head" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        f = self.backbone(x)
        return self.age_head(f).squeeze(-1), self.gen_head(f).squeeze(-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        age, g = self.forward(x)
        age = torch.clamp(age, 0.0, 100.0)
        prob = torch.sigmoid(g)
        return age, prob


class EMA:
    """Exponential Moving Average for the model state dict (shadow copies)."""

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = decay
        self.shadow = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = v.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            sd = model.state_dict()
            for k, v in sd.items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                elif k in self.shadow:
                    self.shadow[k].copy_(v.detach())

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)
