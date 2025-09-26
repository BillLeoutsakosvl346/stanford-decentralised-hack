"""Local simulation runner using Flower's Simulation Engine."""
from __future__ import annotations

import os
from typing import Any, Dict

from flwr.simulation.app import App as SimulationApp, run_simulation

from crowdfl.client.volunteer_client_app import DataSupplier, client_app_factory
from crowdfl.data.medmnist_loader import create_simulation_dataloaders, make_dataloader
from crowdfl.model.tiny_cnn import MODEL_SEED
from crowdfl.server.server_app import app as server_app


def build_data_supplier(dataloaders: Dict[str, Any]) -> DataSupplier:
    def supplier(cid: str, batch_size: int):
        base = dataloaders.get(cid)
        if base is None:
            raise KeyError(f"Unknown client id '{cid}'")
        if base.batch_size == batch_size:
            return base
        return make_dataloader(base.dataset, batch_size=batch_size, seed=MODEL_SEED)

    return supplier


def main() -> None:
    num_clients = int(os.getenv("CROWDFL_SIM_CLIENTS", "10"))
    batch_size = int(os.getenv("CROWDFL_SIM_BATCH", "64"))
    dataset_name = os.getenv("CROWDFL_DATASET", "pathmnist")

    dataloaders = create_simulation_dataloaders(
        num_clients=num_clients,
        batch_size=batch_size,
        name=dataset_name,
        limit_per_client=int(os.getenv("CROWDFL_SIM_LIMIT", "512")),
    )

    client_app = client_app_factory(data_supplier=build_data_supplier(dataloaders))
    sim_app = SimulationApp(server_app=server_app, client_app=client_app)

    run_simulation(
        app=sim_app,
        num_supernodes=num_clients,
        config={
            "num_rounds": int(os.getenv("CROWDFL_NUM_ROUNDS", "5")),
        },
    )


if __name__ == "__main__":
    main()
