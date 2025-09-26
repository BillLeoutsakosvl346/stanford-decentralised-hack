"""Volunteer client consuming packets and reporting updates."""
from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Dict

import torch
from flwr.client import NumPyClient
from flwr.client.app import ClientApp, ClientAppComponents
from flwr.common import Context
from torch.utils.data import DataLoader

from crowdfl.data.medmnist_loader import class_count, load_medmnist_split, make_dataloader
from crowdfl.model.tiny_cnn import MODEL_SEED, build_model
from crowdfl.utils.logging_setup import configure_logging
from crowdfl.utils.params_map import ParamsMap

DataSupplier = Callable[[str, int], DataLoader]


@dataclass
class Packet:
    steps: int
    batch_size: int
    lr: float
    seed: int
    round_idx: int


def _parse_packet(config: Dict) -> Packet:
    steps = int(config.get("steps", 200))
    batch_size = int(config.get("batch_size", 64))
    lr = float(config.get("lr", 0.001))
    seed = int(config.get("seed", MODEL_SEED))
    round_idx = int(config.get("round", 0))
    return Packet(steps=steps, batch_size=batch_size, lr=lr, seed=seed, round_idx=round_idx)


def default_data_supplier(cid: str, batch_size: int) -> DataLoader:
    dataset_name = os.getenv("CROWDFL_DATASET", "pathmnist")
    root = os.getenv("CROWDFL_DATA_ROOT", "./data")
    split = os.getenv("CROWDFL_CLIENT_SPLIT", "train")
    dataset = load_medmnist_split(name=dataset_name, split=split, root=root, download=True)
    seed = MODEL_SEED + abs(hash(cid)) % 1000
    return make_dataloader(dataset, batch_size=batch_size, seed=seed)


class VolunteerNumPyClient(NumPyClient):
    def __init__(self, cid: str, data_supplier: DataSupplier) -> None:
        self.cid = cid
        self.data_supplier = data_supplier
        dataset_name = os.getenv("CROWDFL_DATASET", "pathmnist")
        bundle = build_model(num_classes=class_count(dataset_name))
        self.model = bundle.model
        self.params_map = ParamsMap.capture(self.model)
        self.logger = configure_logging(f"crowdfl.client.{cid}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_loaders: Dict[int, DataLoader] = {}

    def get_parameters(self, config: Dict | None = None):  # noqa: D401
        return self.params_map.model_to_ndarrays(self.model)

    def fit(self, parameters, config):  # noqa: D401
        packet = _parse_packet(config)
        self.params_map.assign_ndarrays(self.model, parameters)
        self._run_packet(packet)
        num_examples = packet.steps * packet.batch_size
        return (
            self.params_map.model_to_ndarrays(self.model),
            int(num_examples),
            {
                "client_id": self.cid,
                "packet_steps": packet.steps,
                "packet_lr": packet.lr,
            },
        )

    def evaluate(self, parameters, config):  # noqa: D401
        # Server performs the authoritative evaluation using the secret validation set.
        return 0.0, 0, {"client_id": self.cid}

    def _run_packet(self, packet: Packet) -> None:
        loader = self._get_loader(packet.batch_size)
        model = self.model.to(self.device)
        model.train()
        torch.manual_seed(packet.seed)
        torch.cuda.manual_seed_all(packet.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=packet.lr)
        criterion = torch.nn.CrossEntropyLoss()
        data_iter = cycle(loader)
        total_loss = 0.0
        for _ in range(packet.steps):
            features, targets = next(data_iter)
            features = features.to(self.device)
            targets = targets.squeeze().long().to(self.device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(packet.steps, 1)
        self.logger.info(
            "Completed packet round=%s steps=%s lr=%s loss=%.4f",
            packet.round_idx,
            packet.steps,
            packet.lr,
            avg_loss,
        )

    def _get_loader(self, batch_size: int) -> DataLoader:
        if batch_size not in self._cached_loaders:
            self._cached_loaders[batch_size] = self.data_supplier(self.cid, batch_size)
        return self._cached_loaders[batch_size]


def client_fn(cid: str, data_supplier: DataSupplier | None = None) -> VolunteerNumPyClient:
    supplier = data_supplier or default_data_supplier
    return VolunteerNumPyClient(cid=cid, data_supplier=supplier)


def client_app_factory(data_supplier: DataSupplier | None = None) -> ClientApp:
    def _client_fn(cid: str) -> VolunteerNumPyClient:
        return client_fn(cid, data_supplier=data_supplier)

    def components(context: Context) -> ClientAppComponents:  # noqa: D401
        return ClientAppComponents(client_fn=_client_fn)

    return ClientApp(components=components)


app = client_app_factory()


__all__ = ["app", "client_app_factory", "client_fn", "VolunteerNumPyClient", "Packet"]
