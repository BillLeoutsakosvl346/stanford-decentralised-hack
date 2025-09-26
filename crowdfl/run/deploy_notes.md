# CrowdFL Deployment Notes

The Flower Deployment Engine makes it painless to move from local simulation to a distributed setup that talks to SuperLink/SuperNodes. This document mirrors the steps from the official Flower guide.

## 0. Prerequisites

- Install Flower CLI tooling:
  ```bash
  pip install "flwr[simulation,deployment]"
  ```
- Make sure the `flower-superlink`, `flower-server-app`, and `flower-client-app` entrypoints are on your PATH (included with Flower 1.5+).

## 1. Start the SuperLink

```bash
flower-superlink --insecure
```

The `--insecure` flag keeps the demo simple; use TLS for production.

## 2. Launch the CrowdFL server

```bash
flower-server-app crowdfl.server.server_app:app
```

Env toggles you might want to tweak:

- `CROWDFL_NUM_ROUNDS` (default `10`)
- `CROWDFL_PACKET_STEPS` (default `300`)
- `CROWDFL_PACKET_BATCH` (default `64`)
- `CROWDFL_PACKET_LR` (default `0.001`)
- `CROWDFL_STATE_DIR` (default `./state`)

Example:

```bash
CROWDFL_NUM_ROUNDS=20 CROWDFL_PACKET_STEPS=200 \
  flower-server-app crowdfl.server.server_app:app
```

## 3. Volunteer clients / SuperNodes

On each contributor machine (or separate terminals when demoing), run:

```bash
flower-client-app crowdfl.client.volunteer_client_app:app
```

If you want to print the payload sizes per round, switch to the wrapped app:

```bash
flower-client-app crowdfl.client.arraysize_mod_wrap:app
```

Clients respect the following environment variables:

- `CROWDFL_DATASET` (`pathmnist` by default)
- `CROWDFL_DATA_ROOT` (`./data`)
- `CROWDFL_CLIENT_SPLIT` (`train`)

## 4. Dashboard

In parallel, run the lightweight web API + static dashboard:

```bash
uvicorn web.api:app --host 0.0.0.0 --port 8000
```

Then open `web/static/index.html` in your browser. The dashboard polls the API every two seconds and shows packet progress plus the current credit leaderboard.

Happy federating!
