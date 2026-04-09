#!/usr/bin/env python3
"""
Batch teacher-label multiple datasets and emit a single meta-training JSONL.

Each manifest entry loads a Hugging Face dataset, extracts a text field, runs:
  1) all experts -> individual_results
  2) binary label from max expert P(unsafe) (same Q features as meta_classifier; no external teacher)

Output rows are compatible with `python -m meta_classifier.train_meta`.

Notes:
- Automatically loads `.env` (HF_TOKEN) via `wrappers.env_utils.load_repo_env()`.
- Never logs tokens; output contains only dataset metadata + model outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregator.expert_runner import run_all_safeguards  # noqa: E402
from expert_q_label import label_from_expert_q  # noqa: E402
from wrappers.env_utils import load_repo_env  # noqa: E402


def _as_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip():
            return float(v.strip())
    except Exception:
        return None
    return None


def _match_filter(ex: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """
    Manifest filter:
      {"field": "toxicity", "op": "eq|neq|gt|ge|lt|le|in|contains", "value": ...}
    """
    field = str(flt.get("field", "")).strip()
    op = str(flt.get("op", "eq")).strip().lower()
    val = flt.get("value")

    if not field:
        return True

    ex_val = ex.get(field)

    if op == "eq":
        return ex_val == val
    if op == "neq":
        return ex_val != val
    if op in {"gt", "ge", "lt", "le"}:
        a = _as_float(ex_val)
        b = _as_float(val)
        if a is None or b is None:
            return False
        if op == "gt":
            return a > b
        if op == "ge":
            return a >= b
        if op == "lt":
            return a < b
        return a <= b
    if op == "in":
        if isinstance(val, (list, tuple, set)):
            return ex_val in val
        return False
    if op == "contains":
        if isinstance(ex_val, str) and isinstance(val, str):
            return val in ex_val
        if isinstance(ex_val, (list, tuple, set)):
            return val in ex_val
        return False
    # Unknown op: do not match
    return False


def _entry_passes_filters(entry: Dict[str, Any], ex: Dict[str, Any]) -> bool:
    """
    Optional manifest fields:
      - "filters_all": [filter, ...] (all must pass)
      - "filters_any": [filter, ...] (at least one must pass)
    """
    all_filters = entry.get("filters_all", [])
    any_filters = entry.get("filters_any", [])

    if isinstance(all_filters, list):
        for flt in all_filters:
            if isinstance(flt, dict) and not _match_filter(ex, flt):
                return False

    if isinstance(any_filters, list) and any_filters:
        ok_any = False
        for flt in any_filters:
            if isinstance(flt, dict) and _match_filter(ex, flt):
                ok_any = True
                break
        if not ok_any:
            return False

    return True


def _iter_hf_rows(
    dataset: str,
    config: Optional[str],
    split: str,
    *,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset
    from datasets import get_dataset_config_names

    try:
        # Important: Some HF datasets have a single config literally named "default".
        # Passing config=None is NOT the same as omitting the config argument.
        if config is None:
            ds = load_dataset(dataset, split=split)
        else:
            ds = load_dataset(dataset, config, split=split)
    except ValueError as e:
        msg = str(e).lower()
        if ("config name is missing" in msg or "available configs" in msg) and not config:
            configs = get_dataset_config_names(dataset)
            if len(configs) == 1:
                ds = load_dataset(dataset, configs[0], split=split)
            elif len(configs) > 1:
                raise ValueError(
                    f"Dataset {dataset!r} requires a config. Available configs: {configs}. "
                    f"Set `config` in the manifest entry."
                ) from e
            else:
                raise
        else:
            raise
    n = 0
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        yield ex
        n += 1
        if limit is not None and n >= limit:
            break


def _extract_text(ex: Dict[str, Any], text_field: str) -> Optional[str]:
    t = ex.get(text_field)
    if not isinstance(t, str):
        return None
    t = t.strip()
    return t if t else None


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("datasets"), list):
        return list(data["datasets"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("Manifest must be a JSON list or an object with `datasets: [...]`.")


def _entry_get_dataset_id(entry: Dict[str, Any]) -> str:
    # Support both schemas:
    # - our original: { "dataset": "org/name", ... }
    # - repo manifest: { "hf_id": "org/name", ... }
    v = entry.get("dataset") or entry.get("hf_id") or ""
    return str(v).strip()


def _entry_get_splits(entry: Dict[str, Any]) -> List[str]:
    # Support:
    # - "split": "train"
    # - "splits": ["train","test"]
    if "splits" in entry and isinstance(entry["splits"], list):
        out = [str(s).strip() for s in entry["splits"] if str(s).strip()]
        return out or ["train"]
    s = str(entry.get("split", "train")).strip() or "train"
    return [s]


def _entry_get_limit(entry: Dict[str, Any]) -> Optional[int]:
    # Support:
    # - "limit": 500
    # - "status_in_repo": { "limit_used": 500 }
    if isinstance(entry.get("limit"), (int, float)):
        return int(entry["limit"])
    status = entry.get("status_in_repo")
    if isinstance(status, dict) and isinstance(status.get("limit_used"), (int, float)):
        return int(status["limit_used"])
    return None


def main() -> int:
    load_repo_env()

    ap = argparse.ArgumentParser(description="Batch expert-Q label datasets from a JSON manifest")
    ap.add_argument("--manifest", type=str, required=True, help="Path to JSON manifest")
    ap.add_argument("--out", type=str, required=True, help="Output meta-ready JSONL path")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Unsafe if max expert P(unsafe) among the four heads >= this value",
    )
    ap.add_argument("--require-expert-outputs", action="store_true", help="Skip rows if any expert errors")
    ap.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional cap overriding each entry's limit",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    entries = _load_manifest(manifest_path)
    if not entries:
        raise SystemExit("Manifest is empty.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_text = 0
    skipped_expert = 0
    first_expert_skip: str = ""

    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            dataset = _entry_get_dataset_id(entry)
            if not dataset:
                continue
            cfg_raw = entry.get("config", None)
            if cfg_raw is None:
                config = None
            else:
                config = str(cfg_raw).strip() or None
            text_field = str(entry.get("text_field", "text")).strip() or "text"

            splits = _entry_get_splits(entry)
            base_limit = _entry_get_limit(entry)
            if args.max_per_dataset is not None:
                base_limit = args.max_per_dataset if base_limit is None else min(base_limit, args.max_per_dataset)

            # Prefer explicit source; else use the richer `hf_id` if present; else dataset.
            source_default = str(entry.get("hf_id", "")).strip() or dataset
            source = str(entry.get("source", source_default)).strip() or source_default
            domain = str(entry.get("domain", "")).strip()
            language = str(entry.get("language", "")).strip()

            for split in splits:
                per_split_limit = base_limit

                for ex in _iter_hf_rows(dataset, config, split, limit=per_split_limit):
                    if not _entry_passes_filters(entry, ex):
                        continue
                    text = _extract_text(ex, text_field)
                    if not text:
                        skipped_text += 1
                        continue

                    individual_results = run_all_safeguards(text)
                    if args.require_expert_outputs:
                        bad = False
                        for ename in ("jailbreak", "toxicity", "sexual", "factuality"):
                            r = individual_results.get(ename, {})
                            if not isinstance(r, dict) or "error" in r:
                                bad = True
                                break
                        if bad:
                            skipped_expert += 1
                            if not first_expert_skip:
                                for ename in ("jailbreak", "toxicity", "sexual", "factuality"):
                                    r = individual_results.get(ename, {})
                                    if isinstance(r, dict) and "error" in r:
                                        first_expert_skip = f"{ename}: {r.get('error')}"
                                        break
                            continue

                    label, verdict, max_q = label_from_expert_q(individual_results, threshold=args.threshold)

                    row: Dict[str, Any] = {
                        "text": text,
                        "label": label,
                        "individual_results": individual_results,
                        "label_source": "expert_q_heuristic",
                        "label_verdict": verdict,
                        "label_threshold": float(args.threshold),
                        "max_expert_q": float(max_q),
                        "dataset": dataset,
                        "config": config,
                        "split": split,
                        "source": source,
                    }
                    if domain:
                        row["domain"] = domain
                    if language:
                        row["language"] = language

                    # carry over a stable id if available
                    for kid in ("id", "idx"):
                        if kid in ex and ex[kid] is not None:
                            row["id"] = str(ex[kid])
                            break

                    f.write(json.dumps({k: v for k, v in row.items() if v is not None}, ensure_ascii=False) + "\n")
                    written += 1

    print(
        f"Wrote {written} rows to {out_path}. Skipped: text={skipped_text}, expert={skipped_expert}"
    )
    if written == 0:
        print(
            "error: no rows written. If expert skips dominated, drop --require-expert-outputs temporarily.",
            file=sys.stderr,
        )
        if first_expert_skip:
            print(f"first expert failure: {first_expert_skip}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

