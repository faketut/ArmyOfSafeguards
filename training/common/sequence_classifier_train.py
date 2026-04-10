from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 42
DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_lora_target_modules(model) -> List[str]:
    names = [n for n, _ in model.named_modules()]

    def _has(fragment: str) -> bool:
        frag = "." + fragment
        return any(frag in n for n in names)

    if _has("query_proj") and _has("value_proj"):
        return ["query_proj", "value_proj"]
    if _has("query") and _has("value"):
        return ["query", "value"]
    if _has("q_proj") and _has("v_proj"):
        return ["q_proj", "v_proj"]

    candidates = ["in_proj", "out_proj", "dense"]
    found = [c for c in candidates if _has(c)]
    return found


def _parse_label(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"unsafe", "harmful", "1", "true", "yes"} else 0
    raise ValueError(f"bad label: {v!r}")


def _load_teacher_jsonl(path: Path, *, domain: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dom = domain.strip().lower()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if dom and str(r.get("domain", "")).strip().lower() != dom:
                continue
            rows.append(r)
    return rows


def _to_dataset(rows: List[Dict[str, Any]], *, text_field: str, label_field: str) -> Dataset:
    texts: List[str] = []
    labels: List[int] = []
    for r in rows:
        t = r.get(text_field)
        if not isinstance(t, str) or not t.strip():
            continue
        try:
            y = _parse_label(r.get(label_field))
        except Exception:
            continue
        texts.append(t.strip())
        labels.append(int(y))
    return Dataset.from_dict({"text": texts, "label": labels})


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if labels is None:
            loss = outputs.loss
        else:
            labels = labels.long()
            if self._class_weights is not None:
                cw = self._class_weights.to(device=logits.device, dtype=logits.dtype)
                loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    labels = np.asarray(labels)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    out: Dict[str, float] = {}
    if len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    out["f1"] = float(f1_score(labels, preds))
    out["pos_rate"] = float(labels.mean())
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fine-tune binary sequence-classification expert (native JSONL or HF manifest)")
    ap.add_argument("--data", type=str, default="", help="JSONL with text+label (+ optional domain filter)")
    ap.add_argument(
        "--hf-manifest",
        type=str,
        default="",
        help="JSON manifest with HF dataset entries (native labels via build_meta_from_hf_labels mapping).",
    )
    ap.add_argument("--domain", type=str, default="toxicity", help="Domain filter for --data rows (default: toxicity)")
    ap.add_argument("--text-field", type=str, default="text")
    ap.add_argument("--label-field", type=str, default="label")
    ap.add_argument("--model-name", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--use-fast-tokenizer", action="store_true")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--output-dir", type=str, default="experts/artifacts/toxicity_ft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--class-weight", type=str, choices=["none", "balanced"], default="balanced")
    ap.add_argument("--lr-scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--target-train-pos-rate", type=float, default=0.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def run_training(args: argparse.Namespace) -> int:
    if args.target_train_pos_rate and float(args.target_train_pos_rate) > 0.0 and args.class_weight == "balanced":
        print("warning: both --target-train-pos-rate and --class-weight balanced set; overriding class-weight -> none")
        args.class_weight = "none"

    if args.fp16 and args.bf16:
        raise SystemExit("Choose only one: --fp16 or --bf16")

    _seed_everything(int(args.seed))

    dom = str(args.domain or "").strip().lower()

    rows: List[Dict[str, Any]] = []
    if args.hf_manifest.strip():
        from datasets import load_dataset

        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root))
        from training.meta.build_meta_from_hf_labels import _extract_text, _map_label  # type: ignore

        manifest = json.loads(Path(args.hf_manifest).read_text(encoding="utf-8"))
        entries = manifest.get("datasets", manifest) if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list):
            raise SystemExit("--hf-manifest must be a JSON list or an object with datasets: [...]")

        for ent in entries:
            if not isinstance(ent, dict):
                continue
            if str(ent.get("domain", "")).strip().lower() != dom:
                continue
            dataset = str(ent.get("hf_id", "") or ent.get("dataset", "")).strip()
            if not dataset:
                continue
            cfg_raw = ent.get("config", None)
            cfg = None if cfg_raw is None else (str(cfg_raw).strip() or None)
            splits = ent.get("splits") or ent.get("split") or ["train"]
            if isinstance(splits, str):
                splits = [splits]
            if not isinstance(splits, list) or not splits:
                splits = ["train"]

            limit = None
            status = ent.get("status_in_repo")
            if isinstance(ent.get("limit"), (int, float)):
                limit = int(ent["limit"])
            elif isinstance(status, dict) and isinstance(status.get("limit_used"), (int, float)):
                limit = int(status["limit_used"])

            for sp in splits:
                sp = str(sp).strip() or "train"
                ds = (
                    load_dataset(dataset, cfg, split=sp, trust_remote_code=True)
                    if cfg
                    else load_dataset(dataset, split=sp, trust_remote_code=True)
                )
                n = 0
                for ex in ds:
                    if limit is not None and n >= limit:
                        break
                    if not isinstance(ex, dict):
                        continue
                    text = _extract_text(dataset, ex, str(ent.get("text_field", "") or ""))
                    if not text:
                        continue
                    unsafe = _map_label(dataset, ex, split_hint=sp)
                    if unsafe is None:
                        continue
                    rows.append({"text": text, "label": "unsafe" if unsafe else "safe"})
                    n += 1

        if not rows:
            raise SystemExit("no rows built from --hf-manifest (check domain filter and mapping)")
        ds_all = _to_dataset(rows, text_field="text", label_field="label")
    else:
        if not args.data.strip():
            raise SystemExit("Provide --data or --hf-manifest")
        rows = _load_teacher_jsonl(Path(args.data), domain=args.domain)
        if not rows:
            raise SystemExit("no rows loaded for domain")
        ds_all = _to_dataset(rows, text_field=args.text_field, label_field=args.label_field)

    if len(ds_all) < 100:
        raise SystemExit(f"not enough rows after filtering: {len(ds_all)}")

    ds_all = ds_all.class_encode_column("label")
    split = ds_all.train_test_split(
        test_size=1.0 - float(args.train_ratio),
        seed=int(args.seed),
        stratify_by_column="label",
    )
    ds = DatasetDict(train=split["train"], valid=split["test"])

    if args.target_train_pos_rate and float(args.target_train_pos_rate) > 0.0:
        r = float(args.target_train_pos_rate)
        if not (0.0 < r < 1.0):
            raise SystemExit("--target-train-pos-rate must be in (0,1)")
        y = np.asarray(ds["train"]["label"], dtype=np.int64)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0:
            raise SystemExit("train split has 0 positives; cannot resample")
        neg_keep = int(np.floor(len(pos_idx) * (1.0 - r) / r))
        neg_keep = max(0, min(neg_keep, len(neg_idx)))
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(neg_idx)
        keep_idx = np.concatenate([pos_idx, neg_idx[:neg_keep]])
        keep_idx = rng.permutation(keep_idx)
        ds = DatasetDict(train=ds["train"].select(keep_idx.tolist()), valid=ds["valid"])

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=bool(args.use_fast_tokenizer))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "LABEL_0", 1: "LABEL_1"},
        label2id={"LABEL_0": 0, "LABEL_1": 1},
    )

    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:
            raise SystemExit("LoRA requested but `peft` is not installed. Install with: pip install peft") from e

        target_modules = _pick_lora_target_modules(model)
        if not target_modules:
            raise SystemExit(
                "LoRA could not find any target_modules in this backbone. "
                "Try without --lora, or extend _pick_lora_target_modules for this model."
            )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tok(batch["text"], truncation=True, max_length=int(args.max_length))

    ds_tok = ds.map(_tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    class_weights = None
    if args.class_weight == "balanced":
        y = np.asarray(ds["train"]["label"], dtype=np.int64)
        pos = float(y.sum())
        neg = float(len(y) - pos)
        w0 = (pos + neg) / max(1.0, 2.0 * neg)
        w1 = (pos + neg) / max(1.0, 2.0 * pos)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    out_dir = Path(args.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    def _make_training_args() -> TrainingArguments:
        steps_per_epoch = int(np.ceil(len(ds_tok["train"]) / max(1, int(args.batch) * int(args.grad_accum))))
        total_steps = int(np.ceil(steps_per_epoch * float(args.epochs)))
        warmup_steps = int(np.floor(max(0.0, min(1.0, float(args.warmup_ratio))) * total_steps))

        common = dict(
            output_dir=str(out_dir),
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            per_device_train_batch_size=int(args.batch),
            per_device_eval_batch_size=int(args.batch),
            gradient_accumulation_steps=int(args.grad_accum),
            num_train_epochs=float(args.epochs),
            lr_scheduler_type=str(args.lr_scheduler),
            warmup_steps=int(warmup_steps),
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            logging_steps=50,
            seed=int(args.seed),
            fp16=bool(args.fp16),
            bf16=bool(args.bf16),
            max_grad_norm=float(args.max_grad_norm),
            report_to=[],
        )
        try:
            return TrainingArguments(evaluation_strategy="epoch", **common)
        except TypeError:
            return TrainingArguments(eval_strategy="epoch", **common)

    targs = _make_training_args()

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        data_collator=collator,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    print(f"[data] train={len(ds_tok['train'])} valid={len(ds_tok['valid'])}")
    print(f"[data] train_pos_rate={float(np.mean(ds['train']['label'])):.4f}")
    trainer.train()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model to: {out_dir}")
    if dom == "jailbreak":
        print("Runtime: set AOS_JAILBREAK_MODEL to this path (or upload and set HF id).")
    elif dom == "toxicity":
        print("Runtime: set AOS_TOXICITY_MODEL to this path (or HF id).")
    elif dom == "factuality":
        print("Runtime: set AOS_FACTUALITY_MODEL to this path (or HF id).")
    elif dom == "sexual":
        print("Runtime: set AOS_SEXUAL_MODEL to this path (or HF id).")
    return 0


def main(argv: Optional[List[str]] = None, **defaults: Any) -> int:
    parser = build_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    return run_training(args)
