"""LoRA (Low-Rank Adaptation) for 2D convolutions.

This module reproduces, byte-for-byte, the adapter behaviour used to train
the champion NPPE-1 checkpoint (``export_resnet18_lora_v6``).

Reference implementation recovered from
``age_gen_model_tl_lora_2-resbet-ver7.ipynb`` (cell 6)::

    class LoRAConv2d(nn.Module):
        def __init__(self, base_conv, r=12, alpha=24.0, dropout=0.05):
            ...
            self.scaling = float(alpha) / float(r)
            self.lora_A = Conv2d(in_channels, r, 1, stride=base_conv.stride, ...)
            self.lora_B = Conv2d(r, out_channels, 1, stride=1, ...)

        def forward(self, x):
            base_out = self.base_conv(x)
            lora_out = self.lora_B(self.drop(self.lora_A(x)))
            if lora_out.shape[2:] != base_out.shape[2:]:
                lora_out = F.interpolate(lora_out, size=base_out.shape[2:], ...)
            return base_out + self.scaling * lora_out

The low-rank update is ``ΔW = (alpha / rank) * (B @ A)`` where ``A`` and
``B`` are 1x1 convolutions.  Note that ``lora_A`` inherits the stride of the
wrapped convolution while ``lora_B`` always uses stride 1, so the adapter
branch preserves the spatial resolution of the frozen base branch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """Wrap a frozen :class:`torch.nn.Conv2d` with a trainable low-rank branch."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        r: int = 12,
        alpha: float = 24.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got {type(base_conv)!r}")

        self.base_conv = base_conv
        for param in self.base_conv.parameters():
            param.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        device = base_conv.weight.device
        dtype = base_conv.weight.dtype
        stride = base_conv.stride

        # A: in_channels -> rank (inherits stride so the two branches align)
        self.lora_A = nn.Conv2d(
            base_conv.in_channels,
            self.r,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        ).to(device=device, dtype=dtype)
        # B: rank -> out_channels (always stride 1)
        self.lora_B = nn.Conv2d(
            self.r,
            base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ).to(device=device, dtype=dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base convolution weight (for compatibility with callers)."""
        return self.base_conv.weight

    @property
    def bias(self):
        return self.base_conv.bias

    def get_effective_weight(self) -> torch.Tensor:
        """Return ``W + (alpha/r) * B @ A`` as an explicit fused kernel.

        Useful for analysis/export; not required for inference because the two
        branches are evaluated separately (exactly as during training).
        """
        with torch.no_grad():
            w = self.base_conv.weight.clone()
            # effective low-rank delta in the same (out, in, 1, 1) layout
            delta = torch.einsum(
                "or,ri->oi", self.lora_B.weight[:, :, 0, 0], self.lora_A.weight[:, :, 0, 0]
            )
            delta = delta[:, :, None, None]
            return w + self.scaling * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.device.type != "cuda" and x.dtype != self.base_conv.weight.dtype:
            x = x.to(self.base_conv.weight.dtype)

        base_out = self.base_conv(x)
        lora_out = self.lora_B(self.drop(self.lora_A(x)))

        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.interpolate(
                lora_out, size=base_out.shape[2:], mode="bilinear", align_corners=False
            )
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:  # noqa: D102
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
