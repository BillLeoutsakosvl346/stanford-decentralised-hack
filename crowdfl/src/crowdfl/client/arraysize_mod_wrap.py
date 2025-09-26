"""Optional client wrapper printing payload sizes."""
from __future__ import annotations

from flwr.client.mod import arrays_size_mod

from crowdfl.client.volunteer_client_app import client_app_factory


def instrumented_client_app():
    base = client_app_factory()
    return arrays_size_mod(base)


app = instrumented_client_app()


__all__ = ["app", "instrumented_client_app"]
