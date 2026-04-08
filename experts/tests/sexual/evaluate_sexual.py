"""Full evaluation of the sexual content expert with accuracy metrics."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.sexual import MODEL_ID, predict

try:
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install datasets scikit-learn")
    sys.exit(1)


def collapse_to_binary(example, label_keys):
    binary_label = 1 if any(example[key] == 1 for key in label_keys) else 0
    return {"true_label": "LABEL_1" if binary_label == 1 else "LABEL_0"}


BENCHMARKS = {
    "x_sensitive": {
        "dataset": "cardiffnlp/x_sensitive",
        "subset": None,
        "split": "test",
        "text_field": "text",
        "label_keys": ["conflictual", "drugs", "profanity", "selfharm", "sex", "spam"],
    }
}


def evaluate_dataset(dataset_name: str, config: Dict[str, Any], limit: int = 100) -> Dict[str, Any]:
    dataset = load_dataset(config["dataset"], split=config["split"], trust_remote_code=True)
    dataset = dataset.map(lambda x: collapse_to_binary(x, config["label_keys"]))

    if len(dataset) > limit:
        dataset = dataset.select(range(limit))

    preds: List[str] = []
    gt: List[str] = []
    for ex in dataset:
        text = ex[config["text_field"]]
        true_label = ex["true_label"]
        out = predict(text)
        preds.append(out["label"])
        gt.append(true_label)

    accuracy = accuracy_score(gt, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, preds, average="weighted", zero_division=0)
    return {
        "dataset": dataset_name,
        "model": MODEL_ID,
        "total_examples": len(preds),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sexual content expert")
    parser.add_argument("--dataset", type=str, choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    datasets_to_run = list(BENCHMARKS.keys()) if args.dataset == "all" else [args.dataset]
    results = [evaluate_dataset(name, BENCHMARKS[name], limit=args.limit) for name in datasets_to_run]
    print(json.dumps(results, indent=2))

    if not args.no_save:
        out = Path(__file__).parent / "evaluation_sexual_results.json"
        out.write_text(
            json.dumps({"timestamp": datetime.now().isoformat(), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

