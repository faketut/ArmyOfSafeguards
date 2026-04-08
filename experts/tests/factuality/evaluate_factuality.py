"""
Evaluation script for factuality expert.

Evaluates on benchmark datasets and calculates accuracy/precision/recall/F1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from experts.factuality import predict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime


BENCHMARKS = {
    "TruthfulQA": {
        "hf_id": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "label_field": "category",
        "label_mapping": {
            "Correct Answers": "LABEL_0",
            "Incorrect Answers": "LABEL_1",
        },
        "default_label": "LABEL_1",
        "note": "Used in training - sanity check only",
    },
    "FEVER": {
        "hf_id": "fever",
        "config": "v1.0",
        "split": "paper_test",
        "text_field": "claim",
        "label_field": "label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",
            "REFUTES": "LABEL_1",
            "NOT ENOUGH INFO": "LABEL_1",
        },
        "default_label": "LABEL_1",
        "note": "Used in training - sanity check only",
    },
    "VitaminC": {
        "hf_id": "tals/vitaminc",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",
            "REFUTES": "LABEL_1",
            "NOT ENOUGH INFO": "LABEL_1",
        },
        "default_label": "LABEL_1",
        "note": "Out-of-distribution - true generalization test",
    },
    "Climate-FEVER": {
        "hf_id": "climate_fever",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "claim_label",
        "label_mapping": {
            "SUPPORTS": "LABEL_0",
            "REFUTES": "LABEL_1",
            "NOT ENOUGH INFO": "LABEL_1",
            "DISPUTED": "LABEL_1",
        },
        "default_label": "LABEL_1",
        "note": "Out-of-distribution - climate-specific claims",
    },
}


def map_label(original_label, label_mapping, default_label):
    if isinstance(original_label, str):
        return label_mapping.get(original_label, default_label)
    return default_label


def evaluate_dataset(name: str, config: dict, limit: int = 100, verbose: bool = True):
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        if "note" in config:
            print(config["note"])
        print(f"{'='*70}")

    hf_id = config["hf_id"]
    ds_config = config.get("config")
    split = config.get("split", "test")

    if ds_config:
        ds = load_dataset(hf_id, ds_config, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)

    subset_size = min(limit, len(ds))
    subset = ds.select(range(subset_size))

    predictions = []
    ground_truth = []
    confidences = []

    label_mapping = config.get("label_mapping", {})
    default_label = config.get("default_label", "LABEL_1")

    for ex in tqdm(subset, disable=not verbose):
        text = ex.get(config.get("text_field", "claim"), str(ex))
        result = predict(text)
        pred = result["label"]
        conf = result["confidence"]

        label_field = config.get("label_field")
        if not label_field or label_field not in ex:
            continue

        gt = map_label(ex[label_field], label_mapping, default_label)
        predictions.append(pred)
        ground_truth.append(gt)
        confidences.append(conf)

    if not predictions:
        return {"dataset": name, "error": "No valid examples with ground truth labels"}

    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary", pos_label="LABEL_0", zero_division=0
    )

    results = {
        "dataset": name,
        "total_examples": len(predictions),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average_confidence": float(sum(confidences) / len(confidences)),
    }

    if verbose:
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-Score: {f1:.2%}")

    return results


def run_all_evaluations(limit: int = 100, save_results: bool = True):
    all_results = []
    for name, cfg in BENCHMARKS.items():
        all_results.append(evaluate_dataset(name, cfg, limit=limit, verbose=True))

    if save_results:
        output_file = Path(__file__).parent / "evaluation_results.json"
        output_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": "ajith-bondili/deberta-v3-factuality-small",
                    "limit_per_dataset": limit,
                    "results": all_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n✅ Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate factuality expert with accuracy metrics")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dataset", type=str, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.dataset:
        res = evaluate_dataset(args.dataset, BENCHMARKS[args.dataset], limit=args.limit)
        if not args.no_save and "error" not in res:
            out = Path(__file__).parent / f"evaluation_{args.dataset.lower()}.json"
            out.write_text(
                json.dumps(
                    {"timestamp": datetime.now().isoformat(), "model": "ajith-bondili/deberta-v3-factuality-small", "result": res},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\n✅ Results saved to: {out}")
    else:
        run_all_evaluations(limit=args.limit, save_results=not args.no_save)

