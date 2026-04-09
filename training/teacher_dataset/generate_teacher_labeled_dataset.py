#!/usr/bin/env python3
"""
Build training rows: expert Q-values + binary label.

For each input text we compute:
  q_i = P(unsafe) for each default expert (jailbreak, toxicity, sexual, factuality),
        same mapping as meta_classifier.feature_builder.

The label is derived in-repo from the experts only:
  unsafe if max(q_1..q_4) >= --threshold, else safe.

Output formats:
  - CSV:  q_jailbreak,q_toxicity,q_sexual,q_factuality,label
  - JSONL: one object per line with q, label, text, individual_results (for meta training)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Repo root
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregator.expert_runner import run_all_safeguards  # noqa: E402
from expert_q_label import label_from_expert_q  # noqa: E402
from meta_classifier.feature_builder import FeatureSpec, build_feature_vector  # noqa: E402
from wrappers.env_utils import load_repo_env  # noqa: E402


def _q_vector(individual_results: Dict[str, Any], *, include_rules: bool) -> Tuple[List[float], List[str]]:
    spec = FeatureSpec(include_bias_features=include_rules)
    names = spec.feature_names()
    vec = build_feature_vector(individual_results, spec=spec)
    return vec, names


def _iter_texts_from_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield i, row


def _load_hf_texts(
    hf_id: str,
    hf_config: Optional[str],
    split: str,
    text_field: str,
    limit: Optional[int],
) -> List[Tuple[Optional[str], str]]:
    from datasets import load_dataset

    if hf_config:
        ds = load_dataset(hf_id, hf_config, split=split)
    else:
        ds = load_dataset(hf_id, split=split)
    out: List[Tuple[Optional[str], str]] = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        t = ex.get(text_field)
        if not isinstance(t, str) or not t.strip():
            continue
        sid = ex.get("id") or ex.get("idx")
        out.append((str(sid) if sid is not None else None, t.strip()))
    return out


def main() -> int:
    load_repo_env()
    ap = argparse.ArgumentParser(description="Expert Q-values + binary label (max-Q threshold)")
    ap.add_argument("--input-jsonl", type=str, default="", help="JSONL with a text field per line")
    ap.add_argument("--text-field", type=str, default="text", help="Field name for input string")
    ap.add_argument("--hf-dataset", type=str, default="", help="HF dataset id (e.g. JailbreakBench)")
    ap.add_argument(
        "--hf-config",
        type=str,
        default="",
        help="Optional HF config / subset name (e.g. behaviors for JBB)",
    )
    ap.add_argument("--hf-split", type=str, default="train", help="Split name (e.g. harmful, benign, train)")
    ap.add_argument("--hf-text-field", type=str, default="Goal", help="Text column for HF rows")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to process")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Label unsafe if max expert P(unsafe) in the four heads >= this value",
    )
    ap.add_argument(
        "--include-rules-q",
        action="store_true",
        help="Append has_rules_tag as 5th Q column (requires rules engine; usually off)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated no-op (same as default: labels always from expert Q heuristic)",
    )
    ap.add_argument(
        "--require-expert-outputs",
        action="store_true",
        help="Exit with error if any expert returns an error dict (no placeholder 0.5 training data)",
    )
    ap.add_argument("--output-csv", type=str, default="", help="Path to CSV q1..q4,label")
    ap.add_argument("--output-jsonl", type=str, default="", help="Path to full JSONL")
    ap.add_argument(
        "--output-meta-jsonl",
        type=str,
        default="",
        help="Path to JSONL suitable for meta_classifier/train_meta.py (individual_results + label)",
    )
    args = ap.parse_args()

    if not args.input_jsonl and not args.hf_dataset:
        ap.error("Provide --input-jsonl or --hf-dataset")

    rows_buffer: List[Dict[str, Any]] = []
    if args.input_jsonl:
        p = Path(args.input_jsonl)
        for _, row in _iter_texts_from_jsonl(p):
            if args.limit is not None and len(rows_buffer) >= args.limit:
                break
            t = row.get(args.text_field)
            if not isinstance(t, str) or not t.strip():
                continue
            rows_buffer.append(dict(row))
    else:
        cfg = args.hf_config or None
        pairs = _load_hf_texts(args.hf_dataset, cfg, args.hf_split, args.hf_text_field, args.limit)
        for sid, t in pairs:
            rows_buffer.append({"id": sid, args.text_field: t})

    if not rows_buffer:
        print("No input rows.", file=sys.stderr)
        return 1

    out_csv = Path(args.output_csv) if args.output_csv else None
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    out_meta = Path(args.output_meta_jsonl) if args.output_meta_jsonl else None

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_meta:
        out_meta.parent.mkdir(parents=True, exist_ok=True)

    include_rules = bool(args.include_rules_q)
    n_q = 4 + (1 if include_rules else 0)

    f_csv = out_csv.open("w", encoding="utf-8", newline="") if out_csv else None
    w_csv: Optional[csv.writer] = None
    if f_csv:
        header = ["q_jailbreak", "q_toxicity", "q_sexual", "q_factuality"]
        if include_rules:
            header.append("q_rules_tag")
        header.append("label")
        w_csv = csv.writer(f_csv)
        w_csv.writerow(header)

    f_j: Any = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    f_m: Any = out_meta.open("w", encoding="utf-8") if out_meta else None

    processed = 0
    expert_error_rows = 0
    q_out: List[float] = []
    for row in rows_buffer:
        text = row.get(args.text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        individual_results = run_all_safeguards(text)
        bad_expert = False
        for ename in ("jailbreak", "toxicity", "sexual", "factuality"):
            er = individual_results.get(ename, {})
            if not isinstance(er, dict) or "error" in er:
                bad_expert = True
                break
        if args.require_expert_outputs and bad_expert:
            expert_error_rows += 1
            continue

        q_full, names = _q_vector(individual_results, include_rules=include_rules)
        if len(q_full) != n_q:
            # Should not happen; keep robust
            q_full = (q_full + [0.0] * n_q)[:n_q]

        lbl_str, verdict, max_q = label_from_expert_q(individual_results, threshold=args.threshold)
        label = 1 if lbl_str == "unsafe" else 0
        teacher_payload: Dict[str, Any] = {
            "label_source": "expert_q_heuristic",
            "verdict": verdict,
            "max_expert_q": max_q,
        }

        q_out = q_full[:4] if not include_rules else q_full

        if w_csv is not None:
            w_csv.writerow([f"{x:.8f}" for x in q_out] + [label])

        record = {
            "q": q_out,
            "q_names": names,
            "label": label,
            "label_str": "unsafe" if label == 1 else "safe",
            "text": text,
            **teacher_payload,
            "individual_results": individual_results,
        }
        # Preserve id if present
        if "id" in row:
            record["id"] = row["id"]

        if f_j:
            f_j.write(json.dumps(record, ensure_ascii=False) + "\n")

        if f_m:
            meta_row = {
                "id": record.get("id"),
                "text": text,
                "label": "unsafe" if label == 1 else "safe",
                "individual_results": individual_results,
                "label_source": teacher_payload.get("label_source"),
            }
            f_m.write(json.dumps({k: v for k, v in meta_row.items() if v is not None}, ensure_ascii=False) + "\n")

        processed += 1

    if f_csv:
        f_csv.close()
    if f_j:
        f_j.close()
    if f_m:
        f_m.close()

    dim = len(q_out) if q_out else (5 if include_rules else 4)
    print(f"Wrote {processed} rows (Q dim={dim}, label from expert_q_heuristic, threshold={args.threshold})")
    if args.require_expert_outputs and expert_error_rows:
        print(f"(skipped {expert_error_rows} input row(s) with missing/failed experts)", file=sys.stderr)
    if args.require_expert_outputs and processed == 0:
        print(
            "error: no rows written — all rows skipped (experts missing, or teacher failed). "
            "Check expert downloads (HF) and network.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
