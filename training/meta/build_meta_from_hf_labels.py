#!/usr/bin/env python3
"""
Build meta-classifier training JSONL from Hugging Face datasets with *native labels*.

For each example:
  - extract a text field (prompt/tweet/comment/...)
  - map dataset label -> binary safe/unsafe
  - run the 4 experts via `run_all_safeguards(text)` to populate `individual_results`

This avoids requiring a teacher model (ShieldGemma / Granite) to label data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from aggregator.expert_runner import run_all_safeguards  # noqa: E402
from wrappers.env_utils import load_repo_env  # noqa: E402


def _as_bool_unsafe(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return bool(int(v) == 1)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"unsafe", "harmful", "toxic", "hate", "yes", "true", "1"}:
            return True
        if s in {"safe", "unharmful", "benign", "no", "false", "0"}:
            return False
    return None


def _map_label(dataset_id: str, ex: Dict[str, Any]) -> Optional[bool]:
    """
    Returns:
      True  -> unsafe
      False -> safe
      None  -> unknown / skip
    """
    did = dataset_id.lower()

    if did == "jackhhao/jailbreak-classification":
        # type: jailbreak|benign
        t = ex.get("type")
        if isinstance(t, str):
            return t.strip().lower() == "jailbreak"
        return None

    if did == "pravalika-9/hate_speech_twitter":
        # labels: 1 hate, 0 non-hate
        return _as_bool_unsafe(ex.get("label"))

    if did == "toxigen/toxigen-data":
        # This HF dataset exposes toxicity scores 1..5; treat >=3 as unsafe.
        # Prefer human score when available, else ai score.
        for k in ("toxicity_human", "toxicity_ai"):
            if k in ex:
                try:
                    return float(ex.get(k)) >= 3.0
                except Exception:
                    pass
        return None

    if did == "allenai/wildguardmix":
        # prompt_harm_label: harmful/unharmful (may be None if no agreement)
        v = ex.get("prompt_harm_label")
        return _as_bool_unsafe(v)

    if did == "cglez/civil_comments_clean":
        # Dataset provides binary int columns (0/1) for toxicity attributes.
        # Treat as unsafe if any of these are 1.
        keys = (
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        )
        any_present = False
        for k in keys:
            if k not in ex:
                continue
            any_present = True
            b = _as_bool_unsafe(ex.get(k))
            if b is True:
                return True
        if any_present:
            return False
        return None

    if did == "cardiffnlp/x_sensitive":
        # Dataset exposes binary indicator columns per category (0/1) and `labels` list.
        # For binary unsafe, treat sex/selfharm as unsafe, others safe.
        for k in ("sex", "selfharm"):
            if k in ex:
                b = _as_bool_unsafe(ex.get(k))
                if b is True:
                    return True
        # If those keys exist and are 0, consider safe.
        if "sex" in ex or "selfharm" in ex:
            return False
        # Fallback: check labels list if present.
        labs = ex.get("labels")
        if isinstance(labs, list):
            labs_l = {str(x).lower() for x in labs}
            return bool({"sex", "selfharm"} & labs_l)
        return None

    # Unknown dataset mapping; try generic
    for k in ("label", "labels", "y", "target"):
        if k in ex:
            return _as_bool_unsafe(ex.get(k))
    return None


def _extract_text(dataset_id: str, ex: Dict[str, Any], text_field: str) -> Optional[str]:
    if text_field:
        t = ex.get(text_field)
        return t.strip() if isinstance(t, str) and t.strip() else None

    did = dataset_id.lower()
    # Common defaults
    candidates: List[str] = []
    if did == "jackhhao/jailbreak-classification":
        candidates = ["prompt", "text"]
    elif did == "pravalika-9/hate_speech_twitter":
        candidates = ["tweet", "text"]
    elif did == "toxigen/toxigen-data":
        candidates = ["text"]
    elif did == "allenai/wildguardmix":
        candidates = ["prompt", "response"]
    elif did == "cardiffnlp/x_sensitive":
        candidates = ["text", "tweet", "post"]
    else:
        candidates = ["text", "prompt", "comment", "tweets", "tweet"]

    for c in candidates:
        t = ex.get(c)
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def main() -> int:
    load_repo_env()
    ap = argparse.ArgumentParser(description="Build meta training JSONL from HF datasets (native labels)")
    ap.add_argument("--dataset", required=True, type=str, help="HF dataset id")
    ap.add_argument("--config", default="", type=str, help="Optional HF config/subset")
    ap.add_argument("--split", default="train", type=str, help="Split name")
    ap.add_argument("--text-field", default="", type=str, help="Optional explicit text field name")
    ap.add_argument("--limit", default=None, type=int, help="Max examples")
    ap.add_argument("--out", required=True, type=str, help="Output JSONL path (meta-ready)")
    ap.add_argument(
        "--require-expert-outputs",
        action="store_true",
        help="Skip examples if any expert errors (recommended to avoid 0.5 placeholders)",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = args.config.strip() or None
    if cfg is None:
        ds = load_dataset(args.dataset, split=args.split)
    else:
        ds = load_dataset(args.dataset, cfg, split=args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_label = 0
    skipped_text = 0
    skipped_expert = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if args.limit is not None and written >= args.limit:
                break

            if not isinstance(ex, dict):
                continue

            text = _extract_text(args.dataset, ex, args.text_field)
            if not text:
                skipped_text += 1
                continue

            unsafe = _map_label(args.dataset, ex)
            if unsafe is None:
                skipped_label += 1
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
                    continue

            row = {
                "text": text,
                "label": "unsafe" if unsafe else "safe",
                "dataset": args.dataset,
                "split": args.split,
                "individual_results": individual_results,
            }
            # keep a stable id if present
            for kid in ("id", "idx"):
                if kid in ex and ex[kid] is not None:
                    row["id"] = str(ex[kid])
                    break

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"Wrote {written} rows to {out_path}. "
        f"Skipped: text={skipped_text}, label={skipped_label}, expert={skipped_expert}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

