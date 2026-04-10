"""
Benchmark stub for factuality expert.

This script outlines how to evaluate the DeBERTa-v3 factuality model
on multiple public factuality datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.factuality import predict
from training.common.hf_datasets import load_hf_split
from tqdm import tqdm


BENCHMARKS = {
    "TruthfulQA": {
        "hf_id": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "label_field": "category",
    },
    "FEVER": {
        "hf_id": "fever",
        "config": "v1.0",
        "split": "test",
        "text_field": "claim",
        "label_field": "label",
    },
    "SciFact": {
        "hf_id": "allenai/scifact",
        "config": "claims",
        "split": "test",
        "text_field": "claim",
        "label_field": "label",
    },
    "VitaminC": {
        "hf_id": "tals/vitaminc",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "label",
    },
    "Climate-FEVER": {
        "hf_id": "climate_fever",
        "config": None,
        "split": "test",
        "text_field": "claim",
        "label_field": "claim_label",
    },
}


def run_benchmark(name: str, config: dict, limit: int = 100, verbose: bool = True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

    try:
        hf_id = config["hf_id"]
        ds_config = config.get("config")
        split = config.get("split", "test")

        ds = load_hf_split(hf_id, ds_config, split)

        subset_size = min(limit, len(ds))
        subset = ds.select(range(subset_size))

        if verbose:
            print(f"Loaded {subset_size} examples from {name}")
            print("Evaluating with factuality expert...\n")

        predictions = []
        confidences = []

        for ex in tqdm(subset, disable=not verbose):
            text_field = config.get("text_field", "claim")
            text = ex.get(text_field, str(ex))
            result = predict(text)
            predictions.append(result["label"])
            confidences.append(result["confidence"])

        avg_confidence = sum(confidences) / len(confidences)
        label_0_count = predictions.count("LABEL_0")
        label_1_count = predictions.count("LABEL_1")

        results = {
            "dataset": name,
            "total_examples": subset_size,
            "predictions": {
                "LABEL_0 (factual)": label_0_count,
                "LABEL_1 (non-factual)": label_1_count,
            },
            "average_confidence": avg_confidence,
            "factual_rate": label_0_count / subset_size,
        }

        if verbose:
            print("\nResults:")
            print(f"  Total examples: {results['total_examples']}")
            print(f"  LABEL_0 (factual): {label_0_count} ({label_0_count/subset_size:.1%})")
            print(f"  LABEL_1 (non-factual): {label_1_count} ({label_1_count/subset_size:.1%})")
            print(f"  Average confidence: {avg_confidence:.2%}")

        return results

    except Exception as e:
        if verbose:
            print(f"⚠️  Could not run {name}: {e}")
        return {"dataset": name, "error": str(e)}


def run_all_benchmarks(limit: int = 100):
    print("=" * 60)
    print("FACTUALITY EXPERT BENCHMARK SUITE")
    print("=" * 60)
    print(f"\nEvaluating on {len(BENCHMARKS)} datasets (limit: {limit} examples each)")

    all_results = []
    for name, config in BENCHMARKS.items():
        all_results.append(run_benchmark(name, config, limit=limit, verbose=True))

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark factuality expert on standard datasets")
    parser.add_argument("--limit", type=int, default=100, help="Maximum examples per dataset (default: 100)")
    parser.add_argument("--dataset", type=str, choices=list(BENCHMARKS.keys()), help="Run single dataset only")
    args = parser.parse_args()

    if args.dataset:
        run_benchmark(args.dataset, BENCHMARKS[args.dataset], limit=args.limit)
    else:
        run_all_benchmarks(limit=args.limit)

