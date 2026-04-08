"""
Shared safeguard runner for aggregators.

This module centralizes the logic for importing and running all available
safeguard experts. Aggregators should depend on this module to avoid drift.
"""

from __future__ import annotations

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Ensure project root is importable when executed as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.registry import get_expert_registry


def run_all_safeguards(text: str) -> Dict[str, Any]:
    """
    Run all available safeguards on the input text.

    Returns:
        Dict keyed by safeguard name. Each value is either:
        - {"label": ..., "confidence": ...} on success, or
        - {"error": "..."} if the safeguard is not importable.
    """
    results: Dict[str, Any] = {}
    registry = get_expert_registry()

    parallel = os.environ.get("AOS_EXPERTS_PARALLEL", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    max_workers = int(os.environ.get("AOS_EXPERTS_MAX_WORKERS", "4") or "4")

    if not parallel:
        for name, spec in registry.items():
            try:
                results[name] = spec.predict(text)
            except ImportError:
                results[name] = {"error": f"{name} expert not available"}
            except Exception as e:
                results[name] = {"error": f"{name} expert failed: {type(e).__name__}: {e}"}
        return results

    # Parallel evaluation (best-effort). Useful for benchmarks; keep off by default.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(spec.predict, text): name for name, spec in registry.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except ImportError:
                results[name] = {"error": f"{name} expert not available"}
            except Exception as e:
                results[name] = {"error": f"{name} expert failed: {type(e).__name__}: {e}"}

    return results


def run_all_safeguards_batch(texts: Sequence[str], *, batch_size: int = 8) -> List[Dict[str, Any]]:
    """
    Batched expert inference for benchmark throughput.

    Returns:
        List[Dict] where each entry is an `individual_results` dict for one text.
    """
    texts_list = list(texts)
    n = len(texts_list)
    if n == 0:
        return []

    registry = get_expert_registry()
    out: List[Dict[str, Any]] = [dict() for _ in range(n)]

    for name, spec in registry.items():
        if spec.predict_batch is None:
            # Fallback to single-item predict
            for i, t in enumerate(texts_list):
                try:
                    out[i][name] = spec.predict(t)
                except Exception as e:
                    out[i][name] = {"error": f"{name} expert failed: {type(e).__name__}: {e}"}
            continue

        # Chunked batch inference
        for start in range(0, n, batch_size):
            chunk = texts_list[start : start + batch_size]
            try:
                preds = spec.predict_batch(chunk)
            except Exception as e:
                preds = [{"error": f"{name} expert failed: {type(e).__name__}: {e}"} for _ in chunk]

            for j, pred in enumerate(preds):
                out[start + j][name] = pred

    return out

