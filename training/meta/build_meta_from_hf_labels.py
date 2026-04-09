#!/usr/bin/env python3
"""
Build meta-classifier training JSONL from Hugging Face datasets with *native labels*.

For each example:
  - extract a text field (prompt/tweet/comment/...)
  - map dataset label -> binary safe/unsafe
  - run the 4 experts via `run_all_safeguards(text)` to populate `individual_results`

This avoids requiring a separate external teacher model to label data.
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


def _map_label(dataset_id: str, ex: Dict[str, Any], split_hint: Optional[str] = None) -> Optional[bool]:
    """
    Returns:
      True  -> unsafe
      False -> safe
      None  -> unknown / skip

    split_hint:
      Used when HF rows do not encode the split name (e.g. JBB-Behaviors harmful vs benign).
    """
    did = dataset_id.lower()

    if "jbb-behaviors" in did or did.endswith("jbb_behaviors"):
        sp = (split_hint or "").strip().lower()
        if sp == "harmful":
            return True
        if sp == "benign":
            return False
        return None

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

    if did == "fever" or did.startswith("fever/"):
        lab = ex.get("label")
        if isinstance(lab, str):
            u = lab.strip().upper()
            if u == "SUPPORTS":
                return False
            if u in {"REFUTES", "NOT ENOUGH INFO"}:
                return True
        return None

    if "vitaminc" in did:
        lab = ex.get("label")
        if isinstance(lab, str):
            u = lab.strip().upper()
            if u == "SUPPORTS":
                return False
            if u in {"REFUTES", "NOT ENOUGH INFO"}:
                return True
        return None

    if "climate_fever" in did:
        lab = ex.get("claim_label") or ex.get("label")
        if isinstance(lab, str):
            u = lab.strip().upper()
            if u == "SUPPORTS":
                return False
            if u in {"REFUTES", "NOT ENOUGH INFO", "DISPUTED"}:
                return True
        return None

    if did == "truthful_qa" or did.endswith("/truthful_qa"):
        cat = ex.get("category")
        if isinstance(cat, str):
            c = cat.strip()
            if c == "Correct Answers":
                return False
            if c == "Incorrect Answers":
                return True
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
    elif "jbb-behaviors" in did or "jailbreakbench" in did:
        candidates = ["Goal", "goal", "text", "prompt"]
    elif did == "fever" or did.startswith("fever/") or "vitaminc" in did or "climate_fever" in did:
        candidates = ["claim", "text", "evidence"]
    else:
        candidates = ["text", "prompt", "comment", "tweets", "tweet"]

    for c in candidates:
        t = ex.get(c)
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def _write_meta_rows(
    f,
    *,
    dataset_id: str,
    split_name: str,
    examples: Iterable[Dict[str, Any]],
    text_field: str,
    limit: Optional[int],
    require_expert_outputs: bool,
) -> Tuple[int, int, int, int]:
    written = 0
    skipped_label = 0
    skipped_text = 0
    skipped_expert = 0
    for i, ex in enumerate(examples):
        if limit is not None and written >= limit:
            break
        if not isinstance(ex, dict):
            continue
        text = _extract_text(dataset_id, ex, text_field)
        if not text:
            skipped_text += 1
            continue
        unsafe = _map_label(dataset_id, ex, split_hint=split_name)
        if unsafe is None:
            skipped_label += 1
            continue
        individual_results = run_all_safeguards(text)
        if require_expert_outputs:
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
            "dataset": dataset_id,
            "split": split_name,
            "source": f"{dataset_id}|{split_name}",
            "individual_results": individual_results,
        }
        for kid in ("id", "idx"):
            if kid in ex and ex[kid] is not None:
                row["id"] = str(ex[kid])
                break
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    return written, skipped_text, skipped_label, skipped_expert


def main() -> int:
    load_repo_env()
    ap = argparse.ArgumentParser(description="Build meta training JSONL from HF datasets (native labels)")
    ap.add_argument("--dataset", default="", type=str, help="HF dataset id (single-dataset mode)")
    ap.add_argument("--config", default="", type=str, help="Optional HF config/subset")
    ap.add_argument("--split", default="train", type=str, help="Split name (single-dataset mode)")
    ap.add_argument("--text-field", default="", type=str, help="Optional explicit text field name")
    ap.add_argument("--limit", default=None, type=int, help="Max examples per dataset/split (single mode or per entry)")
    ap.add_argument("--out", required=True, type=str, help="Output JSONL path (meta-ready)")
    ap.add_argument(
        "--datasets-manifest",
        default="",
        type=str,
        help="JSON file: list or {datasets:[{hf_id, config?, splits?, text_field?, limit?}]} — multi-dataset native meta pool",
    )
    ap.add_argument(
        "--require-expert-outputs",
        action="store_true",
        help="Skip examples if any expert errors (recommended to avoid 0.5 placeholders)",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written_total = 0
    skipped_label = 0
    skipped_text = 0
    skipped_expert = 0

    manifest_path = str(args.datasets_manifest or "").strip()
    jobs: List[Dict[str, Any]] = []
    if manifest_path:
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        entries = raw.get("datasets", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise SystemExit("--datasets-manifest must be a JSON list or {datasets: [...]}")
        for ent in entries:
            if isinstance(ent, dict):
                jobs.append(ent)
    else:
        if not str(args.dataset or "").strip():
            raise SystemExit("Provide --dataset or --datasets-manifest")
        jobs.append(
            {
                "hf_id": args.dataset,
                "config": args.config or None,
                "splits": [args.split],
                "text_field": args.text_field or "",
                "limit": args.limit,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for ent in jobs:
            dataset_id = str(ent.get("hf_id", "") or ent.get("dataset", "")).strip()
            if not dataset_id:
                continue
            cfg_raw = ent.get("config", None)
            cfg = None if cfg_raw is None else str(cfg_raw).strip()
            if cfg == "":
                cfg = None
            splits = ent.get("splits") or ent.get("split") or ["train"]
            if isinstance(splits, str):
                splits = [splits]
            tf = str(ent.get("text_field", "") or "")
            lim = ent.get("limit", None)
            if lim is not None:
                lim = int(lim)

            for sp in splits:
                sp = str(sp).strip() or "train"
                ds = load_dataset(dataset_id, cfg, split=sp, trust_remote_code=True) if cfg else load_dataset(
                    dataset_id, split=sp, trust_remote_code=True
                )
                w, st, sl, se = _write_meta_rows(
                    f,
                    dataset_id=dataset_id,
                    split_name=sp,
                    examples=ds,
                    text_field=tf,
                    limit=lim,
                    require_expert_outputs=bool(args.require_expert_outputs),
                )
                written_total += w
                skipped_text += st
                skipped_label += sl
                skipped_expert += se

    print(
        f"Wrote {written_total} rows to {out_path}. "
        f"Skipped: text={skipped_text}, label={skipped_label}, expert={skipped_expert}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

