"""Server-side secret validation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from crowdfl.data.medmnist_loader import load_medmnist_split, make_dataloader
from crowdfl.model.tiny_cnn import MODEL_SEED


def build_secret_val_loader(
    name: str = "pathmnist",
    batch_size: int = 256,
    root: str | Path = "./data",
    download: bool = True,
    limit: int | None = 2048,
) -> DataLoader:
    dataset = load_medmnist_split(name=name, split="val", root=str(root), download=download)
    if limit is not None:
        indices = list(range(min(len(dataset), limit)))  # type: ignore[arg-type]
        dataset = Subset(dataset, indices)
    return make_dataloader(dataset, batch_size=batch_size, seed=MODEL_SEED)


def evaluate_on_loader(model: nn.Module, dataloader: DataLoader, device: torch.device | None = None) -> Dict[str, float]:
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_acc = 0.0
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.squeeze().long().to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            loss_acc += loss.item() * targets.numel()
    accuracy = correct / total if total else 0.0
    avg_loss = loss_acc / total if total else 0.0
    return {"val_accuracy": accuracy, "val_loss": avg_loss, "val_total": float(total)}


def run_full_evaluation(model: nn.Module, dataloader: DataLoader, device: torch.device | None = None) -> Tuple[float, float]:
    metrics = evaluate_on_loader(model, dataloader, device=device)
    return metrics["val_accuracy"], metrics["val_loss"]


__all__ = [
    "build_secret_val_loader",
    "evaluate_on_loader",
    "run_full_evaluation",
]
