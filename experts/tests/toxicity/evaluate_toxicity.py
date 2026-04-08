"""Full evaluation of the toxicity expert with accuracy metrics."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.toxicity import MODEL_ID, predict

try:
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install datasets scikit-learn")
    sys.exit(1)


BENCHMARKS = {
    "toxigen": {
        "dataset": "toxigen/toxigen-data",
        "subset": "annotated",
        "split": "test",
        "text_field": "text",
        "label_field": "toxicity_human",
        "label_mapping": lambda x: "LABEL_1" if x >= 3 else "LABEL_0",
    },
    "hate_speech18": {
        "dataset": "hate_speech18",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "label_field": "label",
        "label_mapping": {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_1"},
    },
}


def map_label(raw_label: Any, mapping: Any) -> str:
    if callable(mapping):
        return mapping(raw_label)
    return mapping.get(raw_label, "LABEL_0")


def evaluate_dataset(dataset_name: str, config: Dict[str, Any], limit: int = 100) -> Dict[str, Any]:
    if config["subset"]:
        dataset = load_dataset(config["dataset"], config["subset"], split=config["split"], trust_remote_code=True)
    else:
        dataset = load_dataset(config["dataset"], split=config["split"], trust_remote_code=True)

    if len(dataset) > limit:
        dataset = dataset.select(range(limit))

    preds: List[str] = []
    gt: List[str] = []

    for ex in dataset:
        text = ex[config["text_field"]]
        raw = ex[config["label_field"]]
        true_label = map_label(raw, config["label_mapping"])
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
    parser = argparse.ArgumentParser(description="Evaluate toxicity expert")
    parser.add_argument("--dataset", type=str, choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    datasets_to_run = list(BENCHMARKS.keys()) if args.dataset == "all" else [args.dataset]
    results = [evaluate_dataset(name, BENCHMARKS[name], limit=args.limit) for name in datasets_to_run]

    print(json.dumps(results, indent=2))

    if not args.no_save:
        out = Path(__file__).parent / "evaluation_toxicity_results.json"
        out.write_text(
            json.dumps({"timestamp": datetime.now().isoformat(), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

