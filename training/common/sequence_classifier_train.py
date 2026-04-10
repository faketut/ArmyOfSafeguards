from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
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
_REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_METRICS_SCHEMA_VERSION = 1
DEFAULT_SFT_METRICS_REGISTRY = "training/experts/sft_metrics.jsonl"


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


def _load_labeled_jsonl(path: Path, *, domain: str) -> List[Dict[str, Any]]:
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


def _json_safe_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, bool)):
        return x
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        if hasattr(x, "item"):
            v = x.item()
            return float(v) if isinstance(v, (int, float)) else str(v)
    except Exception:
        pass
    return str(x)


def _git_head_short() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _best_eval_from_log_history(log_history: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, float], Optional[float]]:
    """
    Pick the eval log entry with highest eval_roc_auc (matches metric_for_best_model).
    Returns (best_epoch, metrics_without_eval_prefix, best_roc_auc).
    """
    best_auc = float("-inf")
    best_entry: Optional[Dict[str, Any]] = None
    for entry in log_history:
        if "eval_roc_auc" not in entry:
            continue
        try:
            auc = float(entry["eval_roc_auc"])
        except (TypeError, ValueError):
            continue
        if math.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_entry = entry
    if best_entry is None:
        # e.g. single-class validation — take last entry with any eval_* keys
        for entry in reversed(log_history):
            if any(k.startswith("eval_") for k in entry):
                best_entry = entry
                break
    if best_entry is None:
        return None, {}, None
    ep = best_entry.get("epoch")
    best_epoch = float(ep) if isinstance(ep, (int, float)) else None
    metrics: Dict[str, float] = {}
    for k, v in best_entry.items():
        if not k.startswith("eval_"):
            continue
        name = k[len("eval_") :]
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if not math.isnan(fv):
                metrics[name] = fv
    out_auc = metrics.get("roc_auc")
    return best_epoch, metrics, out_auc


def _resolve_metrics_registry_path(args: argparse.Namespace) -> Tuple[str, Optional[Path]]:
    """
    Returns (status, resolved_path).
    status: "disabled" | "enabled"
    """
    reg_raw = (getattr(args, "metrics_registry", "") or "").strip()
    if bool(getattr(args, "no_metrics_registry", False)) or not reg_raw:
        return "disabled", None
    reg_path = Path(reg_raw)
    if not reg_path.is_absolute():
        reg_path = _REPO_ROOT / reg_path
    return "enabled", reg_path


def _append_sft_metrics_registry(
    path: Path,
    *,
    args: argparse.Namespace,
    trainer: Any,
    train_n: int,
    valid_n: int,
    train_pos_rate: float,
    valid_pos_rate: float,
) -> None:
    best_epoch, metrics, _ = _best_eval_from_log_history(list(trainer.state.log_history))
    data_mode = "hf_manifest" if str(args.hf_manifest or "").strip() else "jsonl"
    data_path = str(args.hf_manifest).strip() if data_mode == "hf_manifest" else str(args.data).strip()
    record: Dict[str, Any] = {
        "schema_version": SFT_METRICS_SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head_short(),
        "expert_domain": str(args.domain or "").strip().lower(),
        "base_model": str(args.model_name),
        "data": {"mode": data_mode, "path": data_path},
        "split": {
            "train_ratio": float(args.train_ratio),
            "train_n": int(train_n),
            "valid_n": int(valid_n),
            "train_pos_rate": float(train_pos_rate),
            "valid_pos_rate": float(valid_pos_rate),
        },
        "train_config": {
            "epochs": float(args.epochs),
            "lr": float(args.lr),
            "batch": int(args.batch),
            "grad_accum": int(args.grad_accum),
            "max_length": int(args.max_length),
            "seed": int(args.seed),
            "lora": bool(args.lora),
            "class_weight": str(args.class_weight),
            "fp16": bool(args.fp16),
            "bf16": bool(args.bf16),
            "target_train_pos_rate": float(args.target_train_pos_rate or 0.0),
        },
        "output_dir": str(Path(args.output_dir).resolve()),
        "best_epoch": best_epoch,
        "trainer_best_metric": _json_safe_scalar(getattr(trainer.state, "best_metric", None)),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Some filesystems may not support fsync; best-effort only.
            pass


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
    ap.add_argument(
        "--metrics-registry",
        type=str,
        default=DEFAULT_SFT_METRICS_REGISTRY,
        help=f"Append one JSONL record with validation metrics (best epoch by roc_auc). "
        f"Relative paths resolve under repo root. Empty string disables.",
    )
    ap.add_argument(
        "--no-metrics-registry",
        action="store_true",
        help="Do not append to the metrics registry (overrides --metrics-registry).",
    )
    return ap


def run_training(args: argparse.Namespace) -> int:
    if args.target_train_pos_rate and float(args.target_train_pos_rate) > 0.0 and args.class_weight == "balanced":
        print("warning: both --target-train-pos-rate and --class-weight balanced set; overriding class-weight -> none")
        args.class_weight = "none"

    if args.fp16 and args.bf16:
        raise SystemExit("Choose only one: --fp16 or --bf16")

    _seed_everything(int(args.seed))

    dom = str(args.domain or "").strip().lower()
    reg_status, reg_resolved = _resolve_metrics_registry_path(args)
    print(f"[metrics] registry={reg_status} raw={str(getattr(args, 'metrics_registry', ''))!r} resolved={str(reg_resolved) if reg_resolved else ''}")

    rows: List[Dict[str, Any]] = []
    if args.hf_manifest.strip():
        _root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_root))
        from training.common.hf_datasets import load_hf_split  # type: ignore
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
                ds = load_hf_split(dataset, cfg, sp)
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
        rows = _load_labeled_jsonl(Path(args.data), domain=args.domain)
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

    train_n = len(ds_tok["train"])
    valid_n = len(ds_tok["valid"])
    train_pos_rate = float(np.mean(ds["train"]["label"]))
    valid_pos_rate = float(np.mean(ds["valid"]["label"]))
    print(f"[data] train={train_n} valid={valid_n}")
    print(f"[data] train_pos_rate={train_pos_rate:.4f} valid_pos_rate={valid_pos_rate:.4f}")
    trainer.train()

    if reg_resolved is None:
        # Already printed status above.
        pass
    else:
        try:
            _append_sft_metrics_registry(
                reg_resolved,
                args=args,
                trainer=trainer,
                train_n=train_n,
                valid_n=valid_n,
                train_pos_rate=train_pos_rate,
                valid_pos_rate=valid_pos_rate,
            )
            print(f"[metrics] appended run to registry: {reg_resolved}")
        except Exception as e:
            print(f"[metrics] warning: could not append metrics registry: {e}")
            print(traceback.format_exc())

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
