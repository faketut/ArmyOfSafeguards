"""Sexual content safeguard expert powered by DeBERTa-v3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experts.batching import batched_softmax_predict
from experts.runtime import get_torch_device

MODEL_ID = "faketut/x-sensitive-deberta-binary"

_MODEL_CACHE: Tuple[AutoTokenizer, AutoModelForSequenceClassification] | None = None


def _get_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
        model.to(get_torch_device())
        _MODEL_CACHE = (tok, model)
    return _MODEL_CACHE


def predict(text: str) -> Dict[str, float]:
    """Run the sexual content expert on *text* and return the label + confidence."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    tok, model = _get_model()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    confidence, label_id = torch.max(probabilities, dim=-1)

    label = model.config.id2label.get(label_id.item(), str(label_id.item()))
    return {"label": label, "confidence": float(confidence.item())}


def predict_batch(texts: Sequence[str]) -> List[Dict[str, Any]]:
    tok, model = _get_model()
    return batched_softmax_predict(texts=texts, tokenizer=tok, model=model, max_length=128)


def aggregate(predictions: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Majority-vote aggregation across sexual content experts."""
    votes = Counter()
    confidence_totals = defaultdict(float)

    for prediction in predictions:
        label = prediction.get("label")
        if label is None:
            continue
        confidence = float(prediction.get("confidence", 0.0))
        votes[label] += 1
        confidence_totals[label] += confidence

    if not votes:
        raise ValueError("predictions must be a non-empty iterable of mappings with labels")

    winning_label = min(
        votes.keys(),
        key=lambda lbl: (-votes[lbl], -confidence_totals[lbl], str(lbl)),
    )

    winning_votes = votes[winning_label]
    total_predictions = sum(votes.values())
    confidence = confidence_totals[winning_label] / max(1, winning_votes)

    return {
        "label": winning_label,
        "confidence": confidence,
        "votes": winning_votes,
        "total": total_predictions,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sexual content safeguard expert")
    parser.add_argument("text", nargs="?", help="Text snippet to evaluate for sexual/sensitive content")
    return parser


if __name__ == "__main__":
    cli = _build_cli()
    args = cli.parse_args()

    sample_text = args.text or input("Enter text to evaluate: ")
    result = predict(sample_text)

    print("Prediction:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")

