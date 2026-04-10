# Meta aggregator training: labeling spec & JSONL schema

This document defines how to **annotate data** and **format JSONL** for training the logistic meta-classifier (`meta_classifier/train_meta.py`) used by `aggregator/meta_aggregator.py`.

## Goals

- **Binary target**: `safe` vs `unsafe` (ground truth for whether the *input text* should be blocked or escalated under your policy).
- **Meta input**: expert outputs only — `individual_results` from `aggregator/expert_runner.run_all_safeguards(text)` (each expert returns `{"label", "confidence"}`).

## Label semantics (`label` field)

| Value | Meaning |
|-------|---------|
| `safe` / `0` / `false` | Compliant under policy; allow (or route to normal flow). |
| `unsafe` / `1` / `true` | Violates policy; block, escalate, or require intervention. |

Strings are matched case-insensitively in `train_meta.py` (`unsafe`, `harmful`, `1`, `true`, `yes` → positive class).

## Relationship to triage (`needs_review`)

Runtime aggregation (`policy/triage.py`) maps a scalar \(P(\text{unsafe})\) to:

- `unsafe` — score ≥ threshold (after dynamic threshold, if enabled)
- `needs_review` — gray zone \([\)gray_low, gray_high\()\)
- `safe` — otherwise

**Training convention (conservative, matches benchmark code):**

- Treat **`needs_review` as not safe** when converting predictions to binary metrics (same as `evaluation/run_benchmark.py`: `is_safe = verdict == "safe"`).
- For **ground-truth labels**, prefer explicit `safe` / `unsafe`. If a human reviewer would send an item to moderation **without blocking**, you may either:
  - label it `unsafe` if your policy counts review as a positive hit, or
  - label it `safe` and rely on triage at inference — **pick one policy and stay consistent**.

Document your choice in `training/meta/README.md` (or your project docs).

## Recommended fields per JSONL row

Minimum (required for training):

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Input evaluated by experts (often the user message or model response). |
| `label` | string or int | `safe`/`unsafe` or `0`/`1`. |
| `individual_results` | object | Output of `run_all_safeguards(text)` — added by the generator script. |

Optional (strongly recommended):

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable id for deduplication and audits. |
| `domain` | string | e.g. `jailbreak`, `toxicity`, `sexual`, `factuality`, `benign`, `mixed` — used for grouped evaluation. |
| `source` | string | `benchmark:JBB`, `curated`, `production_sample`, etc. — use for **group splits** to reduce template leakage. |
| `language` | string | e.g. `en`, `zh`. |
| `notes` | string | Rationale for borderline labels. |

## JSONL example (before expert featurization)

```json
{"id":"ex-001","text":"How do I bake bread?","label":"safe","domain":"benign","source":"curated"}
```

## JSONL example (after `generate_expert_features.py`)

```json
{
  "id": "ex-001",
  "text": "How do I bake bread?",
  "label": "safe",
  "domain": "benign",
  "source": "curated",
  "individual_results": {
    "jailbreak": {"label": false, "confidence": 0.99},
    "toxicity": {"label": "LABEL_0", "confidence": 0.95},
    "sexual": {"label": "LABEL_0", "confidence": 0.98},
    "factuality": {"label": "LABEL_0", "confidence": 0.9}
  }
}
```

## Sampling guidance

- Include **adversarial / jailbreak / harmful** prompts (positive class).
- Include **hard negatives**: medical, news, or policy discussion that mentions sensitive terms without violating policy (negative class).
- Balance roughly by volume; use **class weights** or focal-style sampling if heavily imbalanced.

## Feature vector (reference)

The meta model uses `meta_classifier/feature_builder.py`: per-expert `p_unsafe_*` derived from `label` + `confidence`, plus optional `has_rules_tag` if `rules.matches` is present. See that file for label→probability mapping per expert.
