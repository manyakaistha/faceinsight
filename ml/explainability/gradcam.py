"""Grad-CAM implementation for age and gender predictions.

Implements gradient-weighted class activation mapping using the final convolutional
layer of the ResNet backbone. For ResNet50, the last convolution is ``layer4.2.conv3``
in the original torchvision architecture; with LoRA wrappers we need to target the
base convolution inside the LoRA module (or the LoRA module itself's base).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models import AgeGenderModel


class GradCAMExplainer:
    def __init__(
        self,
        model: AgeGenderModel,
        target: Literal["age", "gender"] = "age",
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.target = target
        self.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")
        self._layer = None
        self._hooks = []
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer = self._find_target_layer(layer_name)
        if target_layer is None:
            raise RuntimeError("Could not locate target convolution layer for Grad-CAM")
        self._register_hooks(target_layer)

    def _find_target_layer(self, layer_name: Optional[str]) -> Optional[torch.nn.Module]:
        if layer_name:
            # traverse by name
            for n, m in self.model.named_modules():
                if n == layer_name:
                    return m
        # Hook the complete final convolution output, including its LoRA update.
        # Hooking an inner lora_B branch would explain only the adapter path.
        from ..models.lora import LoRAConv2d

        candidates = [
            module
            for name, module in self.model.backbone.named_modules()
            if "layer4" in name and name.endswith(("conv3", "conv2"))
            and isinstance(module, (torch.nn.Conv2d, LoRAConv2d))
        ]
        if candidates:
            return candidates[-1]
        convolutions = [
            module for name, module in self.model.backbone.named_modules()
            if isinstance(module, torch.nn.Conv2d)
            and not name.endswith(("lora_A", "lora_B"))
        ]
        return convolutions[-1] if convolutions else None

    def _register_hooks(self, layer: torch.nn.Module) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()
            if output.requires_grad:
                self._hooks.append(output.register_hook(capture_gradient))

        def capture_gradient(gradient):
            self._gradients = gradient.detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.enable_grad()
    def explain(
        self,
        input_tensor: torch.Tensor,
        return_overlay: bool = True,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Dict:
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        age, g = self.model.forward(input_tensor)
        if self.target == "age":
            score = age.mean()
        else:
            prob = torch.sigmoid(g).mean()
            score = prob
        self.model.zero_grad()
        score.backward(retain_graph=False)
        if self._activations is None or self._gradients is None:
            self.remove_hooks()
            return {"success": False, "error": "Gradients/activations not captured"}
        act = self._activations  # [1, C, H, W]
        grad = self._gradients  # [1, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True).squeeze(1).clamp(min=0)  # [1, H, W]
        cam = F.interpolate(cam.unsqueeze(1), size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False).squeeze(1)
        cam = cam.detach().cpu().numpy()[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = None
        if return_overlay:
            img = input_tensor.detach().cpu().clone().squeeze(0)
            # denorm
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = (img.clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            overlay = (0.6 * img + 0.4 * heatmap_rgb).astype(np.uint8)
        self.remove_hooks()
        return {
            "success": True,
            "heatmap": cam,
            "heatmap_image": Image.fromarray(heatmap_rgb),
            "overlay_image": Image.fromarray(overlay) if overlay is not None else None,
        }
