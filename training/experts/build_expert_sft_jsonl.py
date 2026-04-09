#!/usr/bin/env python3
"""
Build per-expert SFT JSONL from curriculum_native.yaml (or JSON equivalent).

Each line: {"text", "label" (0|1), "domain": "<expert>", "source": "hf_id|split|idx"}

No teacher models. Labels follow the plan:
  - positive_pools: task-specific positives (force or native_match).
  - negative_pools: task-specific negatives (other hazards may be labeled safe for this head).

Use with training/toxicity/train.py --data <out.jsonl> --domain <expert>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from training.meta.build_meta_from_hf_labels import _extract_text, _map_label  # noqa: E402


def _load_curriculum(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise SystemExit("Install PyYAML to use .yaml curriculum: pip install pyyaml") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict) or "experts" not in data:
        raise SystemExit("Curriculum must be a dict with top-level key 'experts'")
    return data


def _load_hf_split(hf_id: str, config: Optional[str], split: str):
    from datasets import load_dataset

    cfg = config.strip() if isinstance(config, str) and config.strip() else None
    if cfg:
        return load_dataset(hf_id, cfg, split=split, trust_remote_code=True)
    return load_dataset(hf_id, split=split, trust_remote_code=True)


def _iter_pool_rows(
    pool: Dict[str, Any],
    *,
    section_positive: bool,
) -> List[Tuple[str, int, str]]:
    """
    Returns list of (text, label_int, source_tag).
    """
    hf_id = str(pool.get("hf_id", "")).strip()
    if not hf_id:
        return []

    config = pool.get("config", None)
    if config is not None and str(config).lower() in {"null", "none", ""}:
        config = None
    elif isinstance(config, str):
        config = config.strip() or None

    splits = pool.get("splits") or pool.get("split") or ["train"]
    if isinstance(splits, str):
        splits = [splits]

    text_field = str(pool.get("text_field", "") or "")
    limit = pool.get("limit")
    lim = int(limit) if limit is not None else None

    label_mode = str(pool.get("label_mode", "force")).strip().lower()
    force_y: Optional[int] = None
    if label_mode == "force":
        if "y" not in pool:
            raise SystemExit(f"force pool needs integer y: {pool!r}")
        force_y = int(pool["y"])
        if force_y not in (0, 1):
            raise SystemExit(f"force y must be 0 or 1: {pool!r}")
    elif label_mode == "native":
        want_unsafe = bool(pool.get("want_unsafe", section_positive))
    else:
        raise SystemExit(f"Unknown label_mode {label_mode!r}")

    out: List[Tuple[str, int, str]] = []
    seen_per_split = 0

    for sp in splits:
        sp = str(sp).strip()
        ds = _load_hf_split(hf_id, config, sp)
        n = 0
        for i, ex in enumerate(ds):
            if lim is not None and len(out) >= lim:
                break
            if not isinstance(ex, dict):
                continue
            text = _extract_text(hf_id, ex, text_field)
            if not text:
                continue

            if label_mode == "force":
                y = int(force_y)  # type: ignore[arg-type]
            else:
                m = _map_label(hf_id, ex)
                if m is None:
                    continue
                if bool(m) != bool(want_unsafe):
                    continue
                y = 1 if m else 0

            tag = f"{hf_id}|{sp}|{i}"
            out.append((text, y, tag))
            n += 1
        seen_per_split += n

    return out


def _resample_pos_rate(
    rows: List[Dict[str, Any]],
    target_rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if not (0.0 < target_rate < 1.0):
        raise SystemExit("target_pos_rate must be in (0,1)")
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    if not pos:
        raise SystemExit("No positive rows to resample")
    r = float(target_rate)
    neg_keep = int(np.floor(len(pos) * (1.0 - r) / r))
    neg_keep = max(0, min(neg_keep, len(neg)))
    rng = np.random.default_rng(seed)
    neg_idx = rng.permutation(len(neg))[:neg_keep]
    kept_neg = [neg[i] for i in neg_idx]
    out = pos + kept_neg
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build expert SFT JSONL from native curriculum")
    ap.add_argument("--curriculum", type=str, default=str(Path(__file__).with_name("curriculum_native.yaml")))
    ap.add_argument("--expert", type=str, required=True, help="Expert name: jailbreak|toxicity|sexual|factuality")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument(
        "--target-train-pos-rate",
        type=float,
        default=0.0,
        help="If in (0,1), downsample negatives to reach this positive rate on the combined set.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    curriculum_path = Path(args.curriculum)
    if not curriculum_path.is_file():
        raise SystemExit(f"Curriculum not found: {curriculum_path}")

    data = _load_curriculum(curriculum_path)
    experts = data.get("experts", {})
    expert = str(args.expert).strip().lower()
    if expert not in experts:
        raise SystemExit(f"Unknown expert {expert!r}. Keys: {list(experts.keys())}")

    spec = experts[expert]
    if not isinstance(spec, dict):
        raise SystemExit("Expert spec must be a mapping")

    pos_pools = spec.get("positive_pools", []) or []
    neg_pools = spec.get("negative_pools", []) or []
    if not isinstance(pos_pools, list) or not isinstance(neg_pools, list):
        raise SystemExit("positive_pools and negative_pools must be lists")

    combined: List[Tuple[str, int, str]] = []
    for p in pos_pools:
        if not isinstance(p, dict):
            continue
        combined.extend(_iter_pool_rows(p, section_positive=True))
    for p in neg_pools:
        if not isinstance(p, dict):
            continue
        combined.extend(_iter_pool_rows(p, section_positive=False))

    rows: List[Dict[str, Any]] = []
    for text, y, tag in combined:
        rows.append(
            {
                "text": text,
                "label": int(y),
                "domain": expert,
                "source": tag,
            }
        )

    if args.target_train_pos_rate and float(args.target_train_pos_rate) > 0.0:
        rows = _resample_pos_rate(rows, float(args.target_train_pos_rate), int(args.seed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos_n = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"Wrote {len(rows)} rows to {out_path} (positives={pos_n}, rate={pos_n/max(1,len(rows)):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
