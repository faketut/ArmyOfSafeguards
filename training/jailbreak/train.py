from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import evaluate

SEED = 42

DEFAULT_DATASET_ID = "TrustAIRLab/in-the-wild-jailbreak-prompts"
DEFAULT_CONFIGS = (
    "jailbreak_2023_05_07",
    "jailbreak_2023_12_25",
    "regular_2023_05_07",
    "regular_2023_12_25",
)
DEFAULT_MODEL = "microsoft/deberta-v3-base"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train jailbreak expert with capped optimizer steps.")
    p.add_argument(
        "--max-steps",
        type=int,
        required=True,
        choices=[500, 1000],
        help="Max optimizer steps (required): 500 or 1000. Training stops when reached.",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        choices=[0.95, 0.99],
        help="Quantile for token-length probe max_length (default: 0.95).",
    )
    p.add_argument(
        "--probe-samples",
        type=int,
        default=2000,
        help="How many training examples to sample for length quantile (default: 2000).",
    )
    p.add_argument(
        "--target-effective-batch",
        type=str,
        default="auto",
        choices=["auto", "32", "64"],
        help="Target effective batch size B*G. auto prefers 64 else 32 (default: auto).",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--output-dir", type=str, default="./deberta_jailbreak")
    p.add_argument("--fp16", action="store_true", help="Force fp16 if CUDA available.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run probing + config selection then exit before training.",
    )
    return p.parse_args()


def _get_device() -> torch.device:
    # Mirror repo-wide convention.
    mode = os.environ.get("AOS_DEVICE", "auto").strip().lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_raw_dataset(seed: int) -> DatasetDict:
    configs = list(DEFAULT_CONFIGS)
    splits = [load_dataset(DEFAULT_DATASET_ID, cfg, split="train") for cfg in configs]
    raw = concatenate_datasets(splits)
    df = raw.to_pandas()
    df = df.drop_duplicates(subset=["prompt"], keep="first")
    raw = Dataset.from_pandas(df, preserve_index=False)

    raw = raw.rename_column("jailbreak", "label")
    raw = raw.remove_columns([c for c in raw.column_names if c not in ["prompt", "label"]])
    raw = raw.map(lambda x: {"label": x["label"]})
    raw = raw.cast_column("label", ClassLabel(names=["not_jailbreak", "jailbreak"]))

    tmp = raw.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
    test_valid = tmp["test"].train_test_split(test_size=0.5, seed=seed, stratify_by_column="label")
    return DatasetDict(train=tmp["train"], valid=test_valid["train"], test=test_valid["test"])


def _estimate_token_length_quantile(
    *,
    tokenizer,
    prompts: Sequence[str],
    quantile: float,
    max_cap: int = 512,
    min_cap: int = 64,
) -> int:
    lens: List[int] = []
    for t in prompts:
        # do not truncate here; we want true lengths up to cap
        ids = tokenizer(t, truncation=False, add_special_tokens=True).get("input_ids", [])
        if ids:
            lens.append(min(len(ids), max_cap))
    if not lens:
        return 384
    q = float(np.quantile(np.array(lens), quantile))
    q_int = int(max(min_cap, min(max_cap, math.ceil(q))))
    return q_int


def _probe_max_batch_size(
    *,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    start: int = 1,
    max_try: int = 256,
) -> int:
    """
    Probe maximum per-device batch size for a single forward pass.

    Uses an exponential ramp-up then binary search. Returns at least 1.
    """
    model.eval()
    model.to(device)

    def _can_run(bs: int) -> bool:
        text = "x " * (max_length // 2)
        batch = [text] * bs
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            with torch.no_grad():
                _ = model(**enc).logits
            return True
        except RuntimeError as e:
            # OOM or similar
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            return False

    lo = max(1, start)
    hi = lo
    while hi < max_try and _can_run(hi):
        lo = hi
        hi = min(max_try, hi * 2)

    if hi == lo:
        return lo

    # if hi is runnable, return it
    if _can_run(hi):
        return hi

    # binary search between lo (ok) and hi (fail)
    left, right = lo, hi
    while left + 1 < right:
        mid = (left + right) // 2
        if _can_run(mid):
            left = mid
        else:
            right = mid
    return left


def _power_of_two_floor(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(math.log2(x)))


def _pick_grad_accum(batch: int, target: str) -> int:
    if target == "32":
        tgt = 32
    elif target == "64":
        tgt = 64
    else:
        # auto
        tgt = 64 if batch < 64 else batch

    if tgt <= batch:
        return 1
    return int(math.ceil(tgt / batch))


def _steps_per_epoch(num_examples: int, batch: int, grad_accum: int) -> int:
    denom = max(1, batch * grad_accum)
    return int(math.ceil(num_examples / denom))


def _num_epochs_upper(max_steps: int, steps_per_epoch: int) -> int:
    return int(math.ceil(max_steps / max(1, steps_per_epoch)) + 1)

def main() -> int:
    args = _parse_args()
    _seed_everything(args.seed)

    device = _get_device()
    print(f"[Config] device={device}")
    print(f"[Config] max_steps={args.max_steps} (optimizer steps)")

    ds = _load_raw_dataset(args.seed)

    try:
        import sentencepiece  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "This training script requires the `sentencepiece` package for DeBERTa tokenization. "
            "Install it with `pip install sentencepiece` and re-run."
        ) from e

    # Prefer a slow tokenizer to avoid fast-conversion issues in some environments.
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    except Exception:
        # DeBERTa-v3-base uses a SentencePiece tokenizer; fall back explicitly.
        from transformers import DebertaV2Tokenizer

        tok = DebertaV2Tokenizer.from_pretrained(args.model_name)

    # ---- length quantile probe (P95/P99) ----
    train_prompts = list(ds["train"]["prompt"])
    if args.probe_samples and len(train_prompts) > args.probe_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_prompts), size=args.probe_samples, replace=False)
        sample_prompts = [train_prompts[i] for i in idx]
    else:
        sample_prompts = train_prompts

    probe_max_length = _estimate_token_length_quantile(
        tokenizer=tok,
        prompts=sample_prompts,
        quantile=float(args.quantile),
        max_cap=512,
        min_cap=64,
    )
    print(f"[Probe] token_length_quantile={args.quantile} -> probe_max_length={probe_max_length}")

    # ---- batch probe (max -> power-of-two floor) ----
    model_probe = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if device.type == "cpu":
        b_max = 4  # keep reasonable on CPU
    else:
        b_max = _probe_max_batch_size(
            tokenizer=tok, model=model_probe, device=device, max_length=probe_max_length, start=1, max_try=256
        )
    per_device_train_batch_size = _power_of_two_floor(b_max)
    print(f"[Probe] max_batch={b_max} -> batch_pow2={per_device_train_batch_size}")

    grad_accum = _pick_grad_accum(per_device_train_batch_size, args.target_effective_batch)
    # clamp grad_accum to avoid extreme slowdowns
    grad_accum = max(1, min(16, grad_accum))
    effective_batch = per_device_train_batch_size * grad_accum
    print(f"[Config] grad_accum={grad_accum} effective_batch={effective_batch} target={args.target_effective_batch}")

    # ---- tokenize full dataset (training uses fixed max_length=384 by design) ----
    train_max_length = 384

    def tok_fn(batch):
        enc = tok(batch["prompt"], truncation=True, max_length=train_max_length)
        enc["labels"] = batch["label"]
        return enc

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    # enforce int64 labels
    import datasets as _datasets

    for split in ds_tok.keys():
        ds_tok[split] = ds_tok[split].cast_column("labels", _datasets.Value("int64"))

    # ---- steps math ----
    train_examples = len(ds_tok["train"])
    steps_epoch = _steps_per_epoch(train_examples, per_device_train_batch_size, grad_accum)
    epochs_upper = _num_epochs_upper(args.max_steps, steps_epoch)
    effective_epochs = args.max_steps / max(1, steps_epoch)

    print(f"[Steps] train_examples={train_examples}")
    print(f"[Steps] steps_per_epoch={steps_epoch}")
    print(f"[Steps] max_steps={args.max_steps} -> num_train_epochs_upper={epochs_upper} effective_epochs~{effective_epochs:.2f}")

    if args.dry_run:
        print("[DryRun] exiting before training.")
        return 0

    # -------------------------
    # CLASS WEIGHTS
    # -------------------------
    y_train = np.array(ds["train"]["label"])
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
    print("[Config] class_weights:", class_weights.tolist())

    # -------------------------
    # METRICS
    # -------------------------
    metrics = {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
        "roc_auc": evaluate.load("roc_auc", "binary"),
    }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": metrics["accuracy"].compute(predictions=preds, references=labels)["accuracy"],
            "f1": metrics["f1"].compute(predictions=preds, references=labels, average="binary")["f1"],
            "precision": metrics["precision"].compute(predictions=preds, references=labels, average="binary")["precision"],
            "recall": metrics["recall"].compute(predictions=preds, references=labels, average="binary")["recall"],
            "roc_auc": metrics["roc_auc"].compute(prediction_scores=probs, references=labels)["roc_auc"],
        }

    # -------------------------
    # MODEL
    # -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # -------------------------
    # CUSTOM TRAINER (weighted CE)
    # -------------------------
    class WeightedCETrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            device = next(model.parameters()).device
            weights = self.class_weights.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

# -------------------------
# COLLATOR
# -------------------------
    collate = DataCollatorWithPadding(tokenizer=tok)

# -------------------------
# TRAINING ARGS
# -------------------------
    eval_steps = max(50, args.max_steps // 10)
    save_steps = max(50, args.max_steps // 10)
    logging_steps = max(10, args.max_steps // 50)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs_upper,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        metric_for_best_model="roc_auc",
        load_best_model_at_end=True,
        gradient_accumulation_steps=grad_accum,
        fp16=(args.fp16 and device.type == "cuda") or (not args.fp16 and torch.cuda.is_available()),
        report_to="none",
        logging_first_step=True,
    )

# -------------------------
# TRAINER
# -------------------------
    trainer = WeightedCETrainer(
        class_weights=class_weights,
        model=model,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        tokenizer=tok,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

# -------------------------
# TRAIN & EVAL
# -------------------------
    trainer.train()
    trainer.evaluate(ds_tok["test"])

# -------------------------
# SAVE BEST MODEL
# -------------------------
    trainer.save_model(os.path.join(args.output_dir, "best"))
    tok.save_pretrained(os.path.join(args.output_dir, "best"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

