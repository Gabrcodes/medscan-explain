"""Model definition: an ImageNet-pretrained backbone (via `timm`) with a fresh
classification head.

Design choices (justify these in the report / Q&A):
  * Transfer learning, not training from scratch — the datasets are small
    (5.8k / 10k images); a from-scratch CNN would overfit badly and cost far more
    compute (sustainability).
  * EfficientNet-B0 as the default — strong accuracy-per-FLOP, ~5.3M params, fits
    comfortably on a single T4/A10G GPU. ResNet-50 is wired up as a comparison
    backbone for the "alternatives considered" requirement.
  * Two-phase fine-tuning — freeze the backbone for `freeze_backbone_epochs` so the
    randomly-initialised head doesn't wreck pretrained features with large gradients,
    then unfreeze and train end-to-end with a smaller backbone LR.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import TrainConfig


class MedScanClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        # `num_classes=0` -> timm returns pooled features; we add our own head so we
        # control dropout and can grab `features` for Grad-CAM if needed.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self._backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    # -- fine-tuning helpers --------------------------------------------------
    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def param_groups(self, head_lr: float, backbone_lr_mult: float):
        """Separate LRs: small for pretrained backbone, full for the new head."""
        return [
            {"params": self.backbone.parameters(), "lr": head_lr * backbone_lr_mult},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    def gradcam_target_layer(self) -> nn.Module:
        """Last spatial conv block — the layer Grad-CAM hooks.

        timm models expose `feature_info`; the simplest robust choice is the module
        feeding the final pooling. For EfficientNet that's `conv_head`/`bn2`; for
        ResNet it's `layer4`. We fall back to the last Conv2d in the module tree.
        """
        name = self._backbone_name.lower()
        if name.startswith("resnet"):
            return self.backbone.layer4[-1]
        if "efficientnet" in name:
            # timm efficientnet: blocks -> conv_head -> bn2 -> act2 -> global_pool
            if hasattr(self.backbone, "conv_head"):
                return self.backbone.conv_head
            return self.backbone.blocks[-1]
        # generic fallback
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("Could not locate a conv layer for Grad-CAM.")
        return last_conv


def build_model(cfg: TrainConfig, pretrained: bool = True) -> MedScanClassifier:
    return MedScanClassifier(cfg.backbone, cfg.num_classes, pretrained=pretrained)
