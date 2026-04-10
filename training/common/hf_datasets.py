"""
Load Hugging Face dataset splits without ``trust_remote_code`` (deprecated / removed in recent ``datasets``).

The Hub dataset ``fever`` still ships a Python loading script on the default revision; use the
Parquet conversion branch instead. On that branch there is no ``v1.0`` builder config — only the
default config — so ``config: v1.0`` in YAML manifests is treated as "load FEVER parquet".
"""
from __future__ import annotations

from typing import Optional

HF_PARQUET_REVISION = "refs/convert/parquet"


def _normalize_config(config: Optional[str]) -> Optional[str]:
    if config is None:
        return None
    s = str(config).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def load_hf_split(hf_id: str, config: Optional[str], split: str):
    """
    Load a single split. Compatible with recent ``datasets`` (no remote code).

    ``fever`` + ``config`` None or ``v1.0``: loads ``fever`` at ``refs/convert/parquet`` (no config arg).
    """
    from datasets import load_dataset

    hid = (hf_id or "").strip()
    if not hid:
        raise ValueError("hf_id is empty")
    sp = (split or "train").strip()
    cfg = _normalize_config(config)

    if hid.lower() == "fever" and (cfg is None or cfg == "v1.0"):
        return load_dataset(hid, split=sp, revision=HF_PARQUET_REVISION)

    if cfg:
        return load_dataset(hid, cfg, split=sp)
    return load_dataset(hid, split=sp)
