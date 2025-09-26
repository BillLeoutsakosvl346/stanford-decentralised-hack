"""Parameter mapping utilities for deterministic Flower ↔ PyTorch conversions."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays


@dataclass(frozen=True)
class ParamsMap:
    """Capture the order of model.state_dict() once and reuse safely."""

    names: tuple[str, ...]

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "ParamsMap":
        state = model.state_dict()
        return cls(names=tuple(state.keys()))

    def model_to_ndarrays(self, model: torch.nn.Module) -> NDArrays:
        arrays: List[np.ndarray] = []
        state = model.state_dict()
        for name in self.names:
            tensor = state[name].detach().cpu()
            arrays.append(tensor.numpy())
        return arrays

    def ndarrays_to_state(self, arrays: Iterable[np.ndarray], model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
        state = model.state_dict()
        updated = OrderedDict()
        for name, array in zip(self.names, arrays, strict=True):
            target = state[name]
            tensor = torch.as_tensor(array, dtype=target.dtype)
            tensor = tensor.view_as(target)
            updated[name] = tensor
        return updated

    def assign_ndarrays(self, model: torch.nn.Module, arrays: Iterable[np.ndarray]) -> None:
        state = self.ndarrays_to_state(arrays, model)
        model.load_state_dict(state, strict=True)

    def assign_parameters(self, model: torch.nn.Module, parameters: Parameters) -> None:
        ndarrays = parameters_to_ndarrays(parameters)
        self.assign_ndarrays(model, ndarrays)

    def model_to_parameters(self, model: torch.nn.Module) -> Parameters:
        ndarrays = self.model_to_ndarrays(model)
        return ndarrays_to_parameters(ndarrays)


__all__ = ["ParamsMap"]
