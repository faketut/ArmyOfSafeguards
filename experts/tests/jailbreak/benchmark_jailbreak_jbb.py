# JBB-Behaviors benchmarking
# !pip install -q torch transformers datasets scikit-learn tqdm safetensors huggingface_hub pandas

import os
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import pandas as pd

# ======================
# 🔧 CONFIG — EDIT HERE
# ======================
REPO_ID = "tommypang04/finetuned-model-jailbrak"  # your fine-tuned model repo
BATCH_SIZE = 16
MAX_LENGTH = 384
DEVICE_MODE = "auto"  # "auto" | "cpu" | "cuda"
SAVE_CSV_PATH = None  # e.g., "preds.csv" or set to None to skip saving


def _select_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if mode == "cuda" else "cpu")


def load_model_and_tokenizer(repo_id: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval().to(device)
    return tok, model


def get_jbb_texts_and_labels() -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Load JBB-Behaviors (subset='behaviors') and return:
      texts  : list[str] -> the 'Goal' field
      labels : list[int] -> harmful=1 (jailbreak), benign=0 (not_jailbreak)
      meta   : list[dict] -> useful metadata for CSV
    """
    ds_harm: Dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    ds_benign: Dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")

    texts, labels, meta = [], [], []

    for row in ds_harm:
        texts.append(row["Goal"])
        labels.append(1)
        meta.append(
            {
                "split": "harmful",
                "Behavior": row.get("Behavior", ""),
                "Category": row.get("Category", ""),
                "Source": row.get("Source", ""),
                "Target": row.get("Target", ""),
            }
        )

    for row in ds_benign:
        texts.append(row["Goal"])
        labels.append(0)
        meta.append(
            {
                "split": "benign",
                "Behavior": row.get("Behavior", ""),
                "Category": row.get("Category", ""),
                "Source": row.get("Source", ""),
                "Target": row.get("Target", ""),
            }
        )

    return texts, labels, meta


@torch.no_grad()
def batched_predict(
    texts: List[str],
    tok: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 384,
) -> Tuple[List[int], List[float]]:
    """
    Returns:
      pred_labels: argmax class ids (0/1)
      pred_conf: confidence of the predicted class (softmax prob of predicted class)
    """
    pred_labels, pred_conf = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Infer"):
        batch_texts = texts[i : i + batch_size]
        enc = tok(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits  # (B, 2)
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1)
        conf = probs.gather(1, ids.unsqueeze(1)).squeeze(1)

        pred_labels.extend(ids.detach().cpu().tolist())
        pred_conf.extend(conf.detach().cpu().tolist())

    return pred_labels, pred_conf


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        target_names=["not_jailbreak (0)", "jailbreak (1)"],
        digits=4,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


device = _select_device(DEVICE_MODE)
print(f"[Info] Using device: {device}")

print("[Info] Loading JBB-Behaviors (behaviors: harmful + benign) ...")
texts, gold, meta = get_jbb_texts_and_labels()
print(f"[Info] Total examples: {len(texts)}")

print(f"[Info] Loading model & tokenizer: {REPO_ID}")
tok, model = load_model_and_tokenizer(REPO_ID, device)

preds, confs = batched_predict(texts, tok, model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

metrics = compute_metrics(gold, preds)
print("\n=== Results ===")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall   : {metrics['recall']:.4f}")
print(f"F1-Score : {metrics['f1']:.4f}")

print("Confusion Matrix [rows=true 0/1, cols=pred 0/1]:")
print(pd.DataFrame(metrics['confusion_matrix'], index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']))

print("\nClassification Report:")
print(metrics["report"])

