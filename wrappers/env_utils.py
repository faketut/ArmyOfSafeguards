from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _parse_env_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        out[k] = v
    return out


def load_repo_env(repo_root: Optional[str | Path] = None) -> Dict[str, str]:
    """
    Load key/value pairs from the repo `.env` into `os.environ` if missing.

    Intentionally minimal: avoids extra dependencies (python-dotenv) and never logs values.

    Returns a dict of variables that were set in this call.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return {}

    try:
        text = env_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    parsed = _parse_env_text(text)
    set_now: Dict[str, str] = {}
    for k, v in parsed.items():
        if k not in os.environ and v != "":
            os.environ[k] = v
            set_now[k] = v

    # Hugging Face tooling recognizes multiple token env var names.
    if "HF_TOKEN" in parsed and "HF_TOKEN" in set_now:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])
        os.environ.setdefault("HF_ACCESS_TOKEN", os.environ["HF_TOKEN"])

    return set_now

