from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


def batched_softmax_predict(
    *,
    texts: Sequence[str],
    tokenizer,
    model,
    max_length: int,
) -> List[Dict[str, Any]]:
    """
    Generic helper for sequence-classification experts.

    Returns list of {"label": ..., "confidence": ...} per text.
    """
    if not texts:
        return []

    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # (B, C)
        probs = torch.softmax(logits, dim=-1)
        conf, ids = torch.max(probs, dim=-1)

    out: List[Dict[str, Any]] = []
    for c, i in zip(conf.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        label = model.config.id2label.get(int(i), str(int(i)))
        out.append({"label": label, "confidence": float(c)})
    return out


def batched_binary_bool_predict(
    *,
    texts: Sequence[str],
    tokenizer,
    model,
    max_length: int,
) -> List[Dict[str, Any]]:
    """
    Specialized helper for 2-class models whose positive class maps to True.
    """
    if not texts:
        return []

    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # (B, 2)
        probs = torch.softmax(logits, dim=-1)
        ids = torch.argmax(probs, dim=-1)
        conf = probs.gather(1, ids.unsqueeze(1)).squeeze(1)

    out: List[Dict[str, Any]] = []
    for c, i in zip(conf.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        out.append({"label": bool(int(i)), "confidence": float(c)})
    return out

