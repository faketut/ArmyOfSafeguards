from __future__ import annotations

import os
from typing import Literal

import torch

DeviceMode = Literal["auto", "cpu", "cuda"]


def get_device_mode() -> DeviceMode:
    v = os.environ.get("AOS_DEVICE", "auto").strip().lower()
    if v in {"cpu", "cuda", "auto"}:
        return v  # type: ignore[return-value]
    return "auto"


def get_torch_device() -> torch.device:
    mode = get_device_mode()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

