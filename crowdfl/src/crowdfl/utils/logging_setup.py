"""Shared logging configuration."""
from __future__ import annotations

import logging
from logging import Logger

_LOGGER_CACHE: dict[str, Logger] = {}


def configure_logging(name: str = "crowdfl", level: int = logging.INFO) -> Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger


__all__ = ["configure_logging"]
