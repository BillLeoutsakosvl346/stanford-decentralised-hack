"""Helpers to build MedMNIST shards for simulation."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Iterable

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from crowdfl.model.tiny_cnn import DEFAULT_NUM_CLASSES, MODEL_SEED

TransformFactory = Callable[[], transforms.Compose]


def _resolve_dataset_class(name: str):
    info = INFO.get(name)
    if info is None:
        raise ValueError(f"Unknown MedMNIST dataset '{name}'")
    class_name = info["python_class"]
    try:
        return getattr(medmnist, class_name)
    except AttributeError as err:
        raise ValueError(f"Dataset class '{class_name}' not found in medmnist") from err


@lru_cache(maxsize=8)
def dataset_info(name: str) -> Dict:
    info = INFO.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset '{name}'")
    return info


def default_transform(name: str, size: int = 64) -> transforms.Compose:
    info = dataset_info(name)
    mean = tuple(float(m) for m in info.get("mean", [0.5, 0.5, 0.5]))
    std = tuple(float(s) for s in info.get("std", [0.5, 0.5, 0.5]))
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_medmnist_split(
    name: str = "pathmnist",
    split: str = "train",
    root: str = "./data",
    download: bool = True,
    transform_factory: TransformFactory | None = None,
) -> Dataset:
    dataset_cls = _resolve_dataset_class(name)
    transform = transform_factory() if transform_factory else default_transform(name)
    return dataset_cls(split=split, root=root, transform=transform, download=download)


def _subset(dataset: Dataset, indices: Iterable[int]) -> Subset:
    indices_list = list(int(i) for i in indices)
    return Subset(dataset, indices_list)


def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    limit_per_client: int | None = None,
    seed: int = MODEL_SEED,
) -> Dict[str, Dataset]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    total_len = len(dataset)  # type: ignore[arg-type]
    indices = np.arange(total_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    shards = np.array_split(indices, num_clients)

    partitions: Dict[str, Dataset] = {}
    for idx, shard in enumerate(shards):
        cid = f"sim_{idx:03d}"
        limited = shard[:limit_per_client] if limit_per_client is not None else shard
        partitions[cid] = _subset(dataset, limited)
    return partitions


def make_dataloader(dataset: Dataset, batch_size: int, seed: int = MODEL_SEED) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def create_simulation_dataloaders(
    num_clients: int,
    batch_size: int,
    name: str = "pathmnist",
    root: str = "./data",
    limit_per_client: int | None = None,
    download: bool = True,
    seed: int = MODEL_SEED,
) -> Dict[str, DataLoader]:
    dataset = load_medmnist_split(name=name, split="train", root=root, download=download)
    partitions = partition_dataset(dataset, num_clients=num_clients, limit_per_client=limit_per_client, seed=seed)
    return {cid: make_dataloader(ds, batch_size=batch_size, seed=seed + idx) for idx, (cid, ds) in enumerate(partitions.items())}


def class_count(name: str = "pathmnist") -> int:
    info = dataset_info(name)
    return int(info.get("n_classes", DEFAULT_NUM_CLASSES))


__all__ = [
    "create_simulation_dataloaders",
    "load_medmnist_split",
    "partition_dataset",
    "make_dataloader",
    "class_count",
]
