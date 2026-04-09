from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
DEFAULT_BASE_MODEL = "distilroberta-base"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            if self._class_weights is not None:
                cw = self._class_weights.to(logits.device)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune toxicity expert on teacher-labeled meta JSONL")
    ap.add_argument("--data", type=str, required=True, help="teacher_all_for_meta.jsonl (or similar)")
    ap.add_argument("--domain", type=str, default="toxicity", help="Domain filter (default: toxicity)")
    ap.add_argument("--text-field", type=str, default="text")
    ap.add_argument("--label-field", type=str, default="label")
    ap.add_argument("--model-name", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--output-dir", type=str, default="experts/artifacts/toxicity_ft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--class-weight", type=str, choices=["none", "balanced"], default="balanced")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    rows = _load_teacher_jsonl(Path(args.data), domain=args.domain)
    if not rows:
        raise SystemExit("no rows loaded for domain")

    ds_all = _to_dataset(rows, text_field=args.text_field, label_field=args.label_field)
    if len(ds_all) < 100:
        raise SystemExit(f"not enough rows after filtering: {len(ds_all)}")

    # Hugging Face datasets only supports `stratify_by_column` for ClassLabel columns.
    ds_all = ds_all.class_encode_column("label")
    split = ds_all.train_test_split(
        test_size=1.0 - float(args.train_ratio),
        seed=int(args.seed),
        stratify_by_column="label",
    )
    ds = DatasetDict(train=split["train"], valid=split["test"])

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "LABEL_0", 1: "LABEL_1"},
        label2id={"LABEL_0": 0, "LABEL_1": 1},
    )

    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tok(batch["text"], truncation=True, max_length=int(args.max_length))

    ds_tok = ds.map(_tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    # class weights (balanced) on training split
    class_weights = None
    if args.class_weight == "balanced":
        y = np.asarray(ds["train"]["label"], dtype=np.int64)
        pos = float(y.sum())
        neg = float(len(y) - pos)
        # Inverse frequency, normalized
        w0 = (pos + neg) / max(1.0, 2.0 * neg)
        w1 = (pos + neg) / max(1.0, 2.0 * pos)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    out_dir = Path(args.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    def _make_training_args() -> TrainingArguments:
        """
        transformers has renamed/deprecated `evaluation_strategy` -> `eval_strategy` in newer versions.
        Keep compatibility by trying the older arg first, then falling back.
        """
        common = dict(
            output_dir=str(out_dir),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch),
            per_device_eval_batch_size=int(args.batch),
            gradient_accumulation_steps=int(args.grad_accum),
            num_train_epochs=float(args.epochs),
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            logging_steps=50,
            seed=int(args.seed),
            fp16=bool(args.fp16),
            report_to=[],
        )
        try:
            return TrainingArguments(evaluation_strategy="epoch", **common)
        except TypeError:
            # Newer transformers
            return TrainingArguments(eval_strategy="epoch", **common)

    targs = _make_training_args()

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    print(f"[data] train={len(ds_tok['train'])} valid={len(ds_tok['valid'])}")
    print(f"[data] train_pos_rate={float(np.mean(ds['train']['label'])):.4f}")
    trainer.train()

    # Save final
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned toxicity expert to: {out_dir}")
    print("To use it at runtime, set:")
    print(f"  AOS_TOXICITY_MODEL={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

