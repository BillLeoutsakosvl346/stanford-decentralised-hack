"""Packet-based Flower strategy with robust aggregation and credits."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch
from flwr.common import FitRes, Metrics, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedMedian

from crowdfl.data.secret_val import evaluate_on_loader
from crowdfl.server.credits_ledger import CreditsLedger
from crowdfl.utils.logging_setup import configure_logging
from crowdfl.utils.params_map import ParamsMap

PacketConfig = Dict[str, Scalar]


@dataclass
class MetricsWriter:
    path: Path
    lock: Lock = field(default_factory=Lock)

    def write(self, payload: Mapping[str, Scalar]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.path.write_text(json.dumps(payload, indent=2))


class PacketStrategy(FedMedian):
    """Custom Flower Strategy handing out packets and minting credits."""

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        params_map: ParamsMap,
        val_loader,
        ledger: CreditsLedger,
        packet_defaults: PacketConfig,
        metrics_writer: MetricsWriter | None = None,
        target_packets: int = 0,
        device: Optional[torch.device] = None,
        fraction_fit: float = 0.3,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_evaluate: float = 0.0,
        **kwargs,
    ) -> None:
        self.logger = configure_logging("crowdfl.packet_strategy")
        self.model_fn = model_fn
        self.params_map = params_map
        self.val_loader = val_loader
        self.ledger = ledger
        self.packet_defaults = packet_defaults
        self.metrics_writer = metrics_writer
        self.target_packets = target_packets
        self.device = device or torch.device("cpu")
        self.processed_packets = 0
        self.last_val_accuracy, self.last_val_loss = self._initial_evaluation()
        self.extra_config_fn = kwargs.pop("on_fit_config_fn", None)
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_evaluate=fraction_evaluate,
            on_fit_config_fn=self._on_fit_config,
            **kwargs,
        )

    def _initial_evaluation(self) -> Tuple[float, float]:
        model = self.model_fn()
        metrics = evaluate_on_loader(model, self.val_loader, device=self.device)
        return metrics["val_accuracy"], metrics["val_loss"]

    def _on_fit_config(self, server_round: int) -> PacketConfig:
        config = dict(self._build_packet_config(server_round))
        if callable(self.extra_config_fn):
            extra = self.extra_config_fn(server_round)
            if extra:
                config.update(extra)
        return {key: self._ensure_scalar(value) for key, value in config.items()}

    def _build_packet_config(self, server_round: int) -> Iterable[Tuple[str, Scalar]]:
        for key, value in self.packet_defaults.items():
            if key == "seed":
                base_seed = int(value)
                yield key, base_seed + server_round
            else:
                yield key, value
        yield "round", server_round

    @staticmethod
    def _ensure_scalar(value: Scalar) -> Scalar:
        if isinstance(value, (int, float, str, bool, bytes)):
            return value
        if isinstance(value, torch.Tensor):
            return value.item()
        return str(value)

    def initialize_parameters(self, client_manager):  # noqa: D401
        model = self.model_fn()
        parameters = self.params_map.model_to_parameters(model)
        return parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> Optional[Tuple[Parameters, Metrics]]:  # noqa: D401
        aggregate = super().aggregate_fit(server_round, results, failures)
        if aggregate is None:
            self.logger.warning("Round %s produced no aggregate", server_round)
            return None

        parameters, metrics = aggregate
        model = self.model_fn()
        ndarrays = parameters_to_ndarrays(parameters)
        self.params_map.assign_ndarrays(model, ndarrays)

        eval_metrics = evaluate_on_loader(model, self.val_loader, device=self.device)
        val_acc = float(eval_metrics["val_accuracy"])
        delta = max(0.0, val_acc - float(self.last_val_accuracy))
        self.last_val_accuracy = val_acc
        self.last_val_loss = float(eval_metrics["val_loss"])

        credited = self._credit_contributors(results, delta, server_round)
        self.processed_packets += len(results)
        metrics = metrics or {}
        metrics.update(
            {
                "val_accuracy": val_acc,
                "val_loss": self.last_val_loss,
                "delta_accuracy": delta,
                "credited_packets": float(len(credited)),
            }
        )
        self._write_metrics_snapshot(server_round, val_acc, delta)
        self.logger.info(
            "Round %s | val_acc=%.4f delta=%.4f packets=%s/%s",
            server_round,
            val_acc,
            delta,
            self.processed_packets,
            self.target_packets or "-",
        )
        return parameters, metrics

    def _credit_contributors(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        delta: float,
        server_round: int,
    ) -> Dict[str, float]:
        if delta <= 0.0:
            return {}
        total_examples = sum(res.num_examples for _, res in results)
        if total_examples <= 0:
            return {}
        awarded: Dict[str, float] = {}
        for client_proxy, fit_res in results:
            share = fit_res.num_examples / total_examples
            credit_amount = delta * share
            client_id = self._extract_client_id(client_proxy, fit_res)
            self.ledger.credit(client_id, credit_amount, round_idx=server_round)
            awarded[client_id] = credit_amount
        return awarded

    @staticmethod
    def _extract_client_id(client_proxy: ClientProxy, fit_res: FitRes) -> str:
        metrics = fit_res.metrics or {}
        client_metric = metrics.get("client_id")
        if isinstance(client_metric, bytes):
            return client_metric.decode("utf-8")
        if client_metric is not None:
            return str(client_metric)
        return str(client_proxy.cid)

    def _write_metrics_snapshot(self, server_round: int, val_acc: float, delta: float) -> None:
        if not self.metrics_writer:
            return
        payload = {
            "round": int(server_round),
            "val_accuracy": round(val_acc, 6),
            "val_loss": round(self.last_val_loss, 6),
            "delta_accuracy": round(delta, 6),
            "processed_packets": int(self.processed_packets),
            "target_packets": int(self.target_packets),
        }
        self.metrics_writer.write(payload)


__all__ = ["PacketStrategy", "MetricsWriter"]
