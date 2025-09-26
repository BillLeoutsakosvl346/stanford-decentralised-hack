# CrowdFL Hackathon Scaffold

CrowdFL is a packet-based federated learning playground designed for hackathons and quick demos. It lets you iterate locally in **Sim Mode** using Flower's Simulation Engine and later point the exact same apps at a Flower Deployment Engine (SuperLink/SuperNodes) federation.

<div align="center">
  <em>Packets in, credits out.</em>
</div>

## 1. Install & bootstrap

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The editable install exposes `crowdfl.server.server_app:app` and `crowdfl.client.volunteer_client_app:app` to Flower.

## 2. Sim Mode (local quick loop)

1. Launch the server + web API metrics feed in one terminal:
   ```bash
   uvicorn web.api:app --reload --port 8000
   ```
2. In another terminal run the simulation:
   ```bash
   python run/simulate.py
   ```
3. Open `web/static/index.html` in your browser. The page polls the FastAPI endpoints and shows progress + top contributors.

The sim uses the same ServerApp/ClientApp code you'll deploy later. It shuffles a `PathMNIST` subset across simulated clients and evaluates on a secret validation split.

## 3. Deployment (Flower SuperLink/SuperNodes)

When you're ready to leave sim mode, follow `run/deploy_notes.md`. The gist:

```bash
# Terminal 1
flower-superlink --insecure

# Terminal 2 (server)
flower-server-app crowdfl.server.server_app:app

# Terminal 3+ (volunteers)
flower-client-app crowdfl.client.volunteer_client_app:app
```

The deployment doc mirrors Flower's official [Deployment Engine guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html).

## 4. Repo layout

```
crowdfl/
  pyproject.toml     # package metadata & dependencies
  README.md          # this file
  src/crowdfl        # Python package
    ...
  web/               # tiny FastAPI + static dashboard
  run/               # sim/deploy helpers
```

Key modules:

- `crowdfl.model.tiny_cnn` – deterministic CNN tuned for CPU demos
- `crowdfl.data.medmnist_loader` – dataset partition helpers for simulation
- `crowdfl.data.secret_val` – creates the server-only validation loader
- `crowdfl.server.packet_strategy` – wraps Flower FedMedian, adds packet config and credit accounting
- `crowdfl.server.credits_ledger` – small JSON ledger backing the leaderboard + payouts mock
- `crowdfl.server.server_app` – wires the Strategy into a ServerApp and logs per-round metrics for the web UI
- `crowdfl.client.volunteer_client_app` – volunteer client app that consumes packets and reports updates
- `crowdfl.client.arraysize_mod_wrap` – helper to wrap the client with Flower's `arrays_size_mod`
- `crowdfl.utils.params_map` – keeps parameter ordering consistent between PyTorch and Flower `Parameters`
- `crowdfl.utils.logging_setup` – shared logging config across server/client
- `web.api` – FastAPI backend serving `/metrics`, `/leaderboard`, `/payouts/mock`
- `web/static/*` – vanilla HTML/CSS/JS dashboard
- `run/simulate.py` – spins up the Flower Simulation Engine with the packaged Server/Client apps
- `run/deploy_notes.md` – copy/paste Flower Deployment Engine commands

## 5. Credits & licenses

- Flower framework – [Apache-2.0](https://github.com/adap/flower)
- MedMNIST dataset – [CC BY 4.0](https://medmnist.com/)

Happy hacking!
