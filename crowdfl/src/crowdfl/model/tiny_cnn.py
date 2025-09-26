"""Deterministic tiny CNN tailored for 64x64 medical tiles."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

MODEL_SEED = 1337
DEFAULT_NUM_CLASSES = 9  # PathMNIST default


def set_torch_seed(seed: int) -> None:
    """Seed all known PRNGs for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable conv block with BN and GELU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depth = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.point = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


class TinyCNN(nn.Module):
    """Tiny CNN with depthwise separable blocks and lightweight head."""

    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 96, stride=1),
            DepthwiseSeparableBlock(96, 128, stride=2),
            DepthwiseSeparableBlock(128, 160, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(96, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


@dataclass(frozen=True)
class ModelBundle:
    """Container bundling a model with the seed that produced it."""

    model: TinyCNN
    seed: int


def build_model(num_classes: int = DEFAULT_NUM_CLASSES, seed: int = MODEL_SEED) -> ModelBundle:
    """Construct a deterministic TinyCNN."""
    set_torch_seed(seed)
    model = TinyCNN(num_classes=num_classes)
    return ModelBundle(model=model, seed=seed)


def count_trainable_parameters(model: nn.Module) -> Tuple[int, float]:
    """Return parameter count and size in megabytes."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024**2)
    return total, size_mb


__all__ = [
    "MODEL_SEED",
    "DEFAULT_NUM_CLASSES",
    "TinyCNN",
    "ModelBundle",
    "build_model",
    "count_trainable_parameters",
]
