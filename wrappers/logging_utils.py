from __future__ import annotations

import logging
import os
from typing import Optional


def _level_from_env(default: str = "WARNING") -> int:
    raw = os.environ.get("AOS_LOG_LEVEL", default).strip().upper()
    return getattr(logging, raw, logging.WARNING)


def configure_logging_once() -> None:
    """
    Configure root logging once.

    Controlled by env:
      - AOS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default WARNING)
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=_level_from_env(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging_once()
    return logging.getLogger(name)

