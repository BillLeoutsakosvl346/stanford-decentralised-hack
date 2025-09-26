"""Flower ServerApp wiring PacketStrategy."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import torch
from flwr.common import Context
from flwr.server.app import ServerApp, ServerAppComponents

from crowdfl.data.medmnist_loader import class_count
from crowdfl.data.secret_val import build_secret_val_loader
from crowdfl.model.tiny_cnn import MODEL_SEED, build_model
from crowdfl.server.credits_ledger import CreditsLedger
from crowdfl.server.packet_strategy import MetricsWriter, PacketStrategy
from crowdfl.utils.logging_setup import configure_logging
from crowdfl.utils.params_map import ParamsMap


def _model_factory(num_classes: int):
    def factory() -> torch.nn.Module:
        bundle = build_model(num_classes=num_classes, seed=MODEL_SEED)
        return deepcopy(bundle.model)

    return factory


def server_fn(context: Context) -> ServerAppComponents:
    configure_logging("crowdfl.server")
    dataset_name = os.getenv("CROWDFL_DATASET", "pathmnist")
    num_classes = class_count(dataset_name)

    secret_loader = build_secret_val_loader(name=dataset_name)

    model_bundle = build_model(num_classes=num_classes, seed=MODEL_SEED)
    params_map = ParamsMap.capture(model_bundle.model)

    state_dir = Path(os.getenv("CROWDFL_STATE_DIR", "state"))
    ledger = CreditsLedger(state_dir / "credits.json")
    metrics_writer = MetricsWriter(state_dir / "metrics.json")

    num_rounds = int(os.getenv("CROWDFL_NUM_ROUNDS", "10"))
    fraction_fit = float(os.getenv("CROWDFL_FRACTION_FIT", "0.3"))
    min_fit_clients = int(os.getenv("CROWDFL_MIN_FIT_CLIENTS", "3"))
    min_available_clients = int(
        os.getenv("CROWDFL_MIN_AVAILABLE_CLIENTS", str(max(min_fit_clients, 3)))
    )

    packet_defaults = {
        "steps": int(os.getenv("CROWDFL_PACKET_STEPS", "300")),
        "batch_size": int(os.getenv("CROWDFL_PACKET_BATCH", "64")),
        "lr": float(os.getenv("CROWDFL_PACKET_LR", "0.001")),
        "seed": int(os.getenv("CROWDFL_PACKET_SEED", str(MODEL_SEED))),
    }

    target_packets = num_rounds * min_fit_clients

    strategy = PacketStrategy(
        model_fn=_model_factory(num_classes),
        params_map=params_map,
        val_loader=secret_loader,
        ledger=ledger,
        packet_defaults=packet_defaults,
        metrics_writer=metrics_writer,
        target_packets=target_packets,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )

    return ServerAppComponents(strategy=strategy, num_rounds=num_rounds)


app = ServerApp(server_fn=server_fn)


__all__ = ["app", "server_fn"]
