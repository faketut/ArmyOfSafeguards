#!/usr/bin/env python3
"""
Build synthetic_meta_train.jsonl for offline testing of meta_classifier/train_meta.py
without loading HuggingFace expert models.

Rows match the schema expected by train_meta.py (individual_results + label).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _unsafe_expert_row(kind: str, strength: float) -> Dict[str, Any]:
    """kind in toxic, jailbreak, sexual, factuality, clean"""
    s = min(max(strength, 0.05), 0.99)
    if kind == "jailbreak":
        return {"label": True, "confidence": s}
    if kind == "toxic":
        return {"label": "unsafe", "confidence": s}
    if kind == "sexual":
        return {"label": "LABEL_1", "confidence": s}
    if kind == "factuality":
        return {"label": "LABEL_1", "confidence": s}
    return {"label": "LABEL_0", "confidence": 0.95}


def _safe_expert_row() -> Dict[str, Any]:
    return {
        "jailbreak": {"label": False, "confidence": 0.97},
        "toxicity": {"label": "LABEL_0", "confidence": 0.96},
        "sexual": {"label": "LABEL_0", "confidence": 0.95},
        "factuality": {"label": "LABEL_0", "confidence": 0.92},
    }


def _unsafe_mixed(strength: float) -> Dict[str, Any]:
    return {
        "jailbreak": _unsafe_expert_row("jailbreak", strength + 0.05),
        "toxicity": _unsafe_expert_row("toxic", strength),
        "sexual": {"label": "LABEL_0", "confidence": 0.9},
        "factuality": {"label": "LABEL_0", "confidence": 0.88},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write synthetic meta training JSONL")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "synthetic_meta_train.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=80, help="Total rows")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows: List[Dict[str, Any]] = []
    for i in range(args.n):
        y = rng.random() < 0.45
        st = rng.uniform(0.55, 0.95)
        if y:
            mode = rng.choice(["mixed", "jb", "tox", "sex", "fact"])
            if mode == "mixed":
                ind = _unsafe_mixed(st)
            elif mode == "jb":
                ind = {
                    "jailbreak": _unsafe_expert_row("jailbreak", st),
                    "toxicity": {"label": "LABEL_0", "confidence": 0.9},
                    "sexual": {"label": "LABEL_0", "confidence": 0.9},
                    "factuality": {"label": "LABEL_0", "confidence": 0.88},
                }
            elif mode == "tox":
                ind = {
                    "jailbreak": {"label": False, "confidence": 0.85},
                    "toxicity": _unsafe_expert_row("toxic", st),
                    "sexual": {"label": "LABEL_0", "confidence": 0.9},
                    "factuality": {"label": "LABEL_0", "confidence": 0.9},
                }
            elif mode == "sex":
                ind = {
                    "jailbreak": {"label": False, "confidence": 0.9},
                    "toxicity": {"label": "LABEL_0", "confidence": 0.9},
                    "sexual": _unsafe_expert_row("sexual", st),
                    "factuality": {"label": "LABEL_0", "confidence": 0.88},
                }
            else:
                ind = {
                    "jailbreak": {"label": False, "confidence": 0.92},
                    "toxicity": {"label": "LABEL_0", "confidence": 0.9},
                    "sexual": {"label": "LABEL_0", "confidence": 0.9},
                    "factuality": _unsafe_expert_row("factuality", st),
                }
            label = "unsafe"
        else:
            ind = _safe_expert_row()
            # occasional noise
            if rng.random() < 0.08:
                ind["toxicity"] = {"label": "LABEL_0", "confidence": 0.65}
            label = "safe"

        rows.append(
            {
                "id": f"syn-{i:04d}",
                "text": f"synthetic placeholder {i}",
                "label": label,
                "domain": rng.choice(["benign", "jailbreak", "toxicity", "mixed"]),
                "source": "synthetic",
                "individual_results": ind,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
