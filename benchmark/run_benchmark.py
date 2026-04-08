"""
Benchmarking pipeline for Army of Safeguards.

This script evaluates the safeguarding system (via aggregator) against
public benchmark datasets for safety and moderation.

For detailed usage instructions, examples, and authentication setup,
see benchmark/README.md
"""

import sys
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aggregator.expert_runner import run_all_safeguards_batch

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Import aggregators - will be dynamically selected
try:
    from aggregator.base_aggregator import evaluate_text as evaluate_text_base
    BASE_AGGREGATOR_AVAILABLE = True
except ImportError:
    BASE_AGGREGATOR_AVAILABLE = False
    evaluate_text_base = None

try:
    from aggregator.weighted_aggregator import evaluate_text as evaluate_text_weighted
    WEIGHTED_AGGREGATOR_AVAILABLE = True
except ImportError:
    WEIGHTED_AGGREGATOR_AVAILABLE = False
    evaluate_text_weighted = None

try:
    from aggregator.meta_aggregator import evaluate_text as evaluate_text_meta
    META_AGGREGATOR_AVAILABLE = True
except ImportError:
    META_AGGREGATOR_AVAILABLE = False
    evaluate_text_meta = None

try:
    from granite_guardian.granite_guardian_wrapper import evaluate_text as evaluate_text_granite
    GRANITE_AVAILABLE = True
except ImportError:
    GRANITE_AVAILABLE = False
    evaluate_text_granite = None

try:
    from shieldgemma.shieldgemma_wrapper import evaluate_text as evaluate_text_shieldgemma
    SHIELDGEMMA_AVAILABLE = True
except ImportError:
    SHIELDGEMMA_AVAILABLE = False
    evaluate_text_shieldgemma = None

# HuggingFace authentication
try:
    from huggingface_hub import login, whoami
    HF_AUTH_AVAILABLE = True
except ImportError:
    HF_AUTH_AVAILABLE = False


# Benchmark dataset configurations
BENCHMARKS = {
    "HarmBench": {
        "hf_id": "walledai/HarmBench",
        "config": "standard",  # Options: 'contextual', 'copyright', 'standard'
        "split": "train",  # HarmBench only has 'train' split
        "text_field": "prompt",  # HarmBench field: prompt (or instructions)
        "label_field": None,  # HarmBench is a red teaming dataset - all examples are harmful/unsafe
        "label_mapping": {
            # Since HarmBench is a red teaming dataset, all examples are considered unsafe
            # If a label field exists, we'll map it, otherwise default to unsafe
        },
        "default_label": "unsafe",  # All HarmBench examples are harmful by design
        "note": "Jailbreak & harmful-content robustness benchmark (using 'standard' config, 'train' split)",
        "requires_auth": True,  # This is a gated dataset
        "available_configs": ["contextual", "copyright", "standard"]  # Available config options
    },
    "JailbreakBench": {
        "hf_id": "JailbreakBench/JBB-Behaviors",
        "config": "behaviors",
        "split": None,  # Has 'harmful' and 'benign' splits
        "text_field": "Goal",
        "label_field": None,  # Label determined by split
        "label_mapping": {
            "harmful": "unsafe",
            "benign": "safe"
        },
        "note": "Jailbreak attempt detection benchmark",
        "requires_auth": False
    },
    "WildGuardMix": {
        "hf_id": "allenai/wildguardmix",
        "config": "wildguardtest",  # Options: 'wildguardtrain', 'wildguardtest'
        "split": None,  # Split is determined by config
        "text_field": "prompt",
        "label_field": "prompt_harm_label",  # Field indicating if prompt is harmful
        "label_mapping": {
            "unharmful": "safe",
            "harmful": "unsafe"
        },
        "note": "Moderation / guardrail benchmark (using 'wildguardtest' config)",
        "requires_auth": True,  # This is a gated dataset
        "available_configs": ["wildguardtrain", "wildguardtest"]  # Available config options
    }
}


def check_hf_authentication() -> bool:
    """
    Check if user is authenticated with HuggingFace.
    
    Returns:
        True if authenticated, False otherwise
    """
    if not HF_AUTH_AVAILABLE:
        return False
    
    try:
        user_info = whoami()
        return user_info is not None
    except Exception:
        return False


def authenticate_hf(token: Optional[str] = None) -> bool:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HuggingFace token (if None, will try to use existing token or prompt)
        
    Returns:
        True if authentication successful, False otherwise
    """
    if not HF_AUTH_AVAILABLE:
        print("⚠️  huggingface_hub not available. Install it with: pip install huggingface_hub")
        return False
    
    try:
        if token:
            # Use provided token
            login(token=token)
        else:
            # Check if token is in environment
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                login(token=hf_token)
            else:
                # Try interactive login
                print("Attempting to authenticate with HuggingFace Hub...")
                login()
        
        # Verify authentication
        if check_hf_authentication():
            print("✅ Successfully authenticated with HuggingFace Hub")
            return True
        else:
            print("⚠️  Authentication verification failed")
            return False
    except Exception as e:
        print(f"⚠️  Authentication error: {e}")
        return False


def load_benchmark_dataset(benchmark_name: str, limit: Optional[int] = None, hf_token: Optional[str] = None, config_override: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Load a benchmark dataset and extract texts and labels.
    
    Args:
        benchmark_name: Name of the benchmark
        limit: Maximum number of examples to load
        hf_token: Optional HuggingFace token for gated datasets
        config_override: Optional config dictionary to override default config
        
    Returns:
        Tuple of (texts, labels, metadata)
    """
    config = config_override if config_override else BENCHMARKS[benchmark_name]
    hf_id = config["hf_id"]
    ds_config = config.get("config")
    split = config.get("split")
    text_field = config["text_field"]
    label_field = config.get("label_field")
    label_mapping = config.get("label_mapping", {})
    requires_auth = config.get("requires_auth", False)
    
    # Check authentication for gated datasets
    if requires_auth:
        if not check_hf_authentication():
            # Try to authenticate
            auth_success = False
            if hf_token:
                auth_success = authenticate_hf(hf_token)
            else:
                # Try to authenticate with existing token or prompt
                auth_success = authenticate_hf()
            
            if not auth_success:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"Authentication required for {benchmark_name}\n"
                    f"{'='*70}\n"
                    f"This is a gated dataset that requires HuggingFace authentication.\n\n"
                    f"Steps to authenticate:\n"
                    f"1. Get a token from: https://huggingface.co/settings/tokens\n"
                    f"2. Accept the dataset terms at: https://huggingface.co/datasets/{hf_id}\n"
                    f"3. Run with token:\n"
                    f"   python benchmark/run_benchmark.py --benchmark {benchmark_name} --hf-token YOUR_TOKEN\n"
                    f"   Or set environment variable:\n"
                    f"   export HF_TOKEN=YOUR_TOKEN  # Linux/Mac\n"
                    f"   set HF_TOKEN=YOUR_TOKEN      # Windows\n"
                    f"{'='*70}"
                )
    
    texts = []
    labels = []
    metadata = []
    
    try:
        # Special handling for JailbreakBench which has multiple splits
        if benchmark_name == "JailbreakBench":
            # Load both harmful and benign splits
            ds_harmful = load_dataset(hf_id, ds_config, split="harmful", trust_remote_code=True)
            ds_benign = load_dataset(hf_id, ds_config, split="benign", trust_remote_code=True)
            
            # Process harmful examples
            for ex in ds_harmful:
                # Try to find text field (case-insensitive)
                text = None
                for key in ex.keys():
                    if key.lower() == text_field.lower():
                        text = ex[key]
                        break
                
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    texts.append(text)
                    labels.append("unsafe")
                    metadata.append({
                        "split": "harmful",
                        "behavior": ex.get("Behavior", ex.get("behavior", "")),
                        "category": ex.get("Category", ex.get("category", "")),
                        **{k: v for k, v in ex.items() if k.lower() != text_field.lower()}
                    })
            
            # Process benign examples
            for ex in ds_benign:
                # Try to find text field (case-insensitive)
                text = None
                for key in ex.keys():
                    if key.lower() == text_field.lower():
                        text = ex[key]
                        break
                
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    texts.append(text)
                    labels.append("safe")
                    metadata.append({
                        "split": "benign",
                        "behavior": ex.get("Behavior", ex.get("behavior", "")),
                        "category": ex.get("Category", ex.get("category", "")),
                        **{k: v for k, v in ex.items() if k.lower() != text_field.lower()}
                    })
        else:
            # Standard dataset loading
            if ds_config:
                try:
                    ds = load_dataset(hf_id, ds_config, split=split, trust_remote_code=True)
                except ValueError as e:
                    # Check if it's a split error
                    if "Unknown split" in str(e) or "Should be one of" in str(e):
                        # Try to get available splits
                        from datasets import get_dataset_config_names, get_dataset_split_names
                        try:
                            available_splits = get_dataset_split_names(hf_id, config_name=ds_config)
                            raise ValueError(
                                f"\n{'='*70}\n"
                                f"Invalid split '{split}' for {benchmark_name}\n"
                                f"{'='*70}\n"
                                f"Available splits: {available_splits}\n"
                                f"Update the 'split' field in BENCHMARKS configuration.\n"
                                f"{'='*70}\n"
                                f"Original error: {str(e)}"
                            )
                        except:
                            raise ValueError(
                                f"\n{'='*70}\n"
                                f"Invalid split '{split}' for {benchmark_name}\n"
                                f"{'='*70}\n"
                                f"Please check the dataset documentation for available splits.\n"
                                f"Update the 'split' field in BENCHMARKS configuration.\n"
                                f"{'='*70}\n"
                                f"Original error: {str(e)}"
                            )
                    raise
            else:
                # Some datasets require a config - try to load with split only
                # If that fails, the error will be caught and reported
                try:
                    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
                except ValueError as e:
                    if "Config name is missing" in str(e) or "config" in str(e).lower():
                        available_configs = config.get("available_configs", [])
                        if available_configs:
                            raise ValueError(
                                f"\n{'='*70}\n"
                                f"Config required for {benchmark_name}\n"
                                f"{'='*70}\n"
                                f"This dataset requires a config name. Available configs: {available_configs}\n"
                                f"Default config '{available_configs[0]}' will be used.\n"
                                f"To use a different config, modify BENCHMARKS in run_benchmark.py\n"
                                f"{'='*70}\n"
                                f"Original error: {str(e)}"
                            )
                    elif "Unknown split" in str(e) or "Should be one of" in str(e):
                        raise ValueError(
                            f"\n{'='*70}\n"
                            f"Invalid split '{split}' for {benchmark_name}\n"
                            f"{'='*70}\n"
                            f"Please check the dataset documentation for available splits.\n"
                            f"Update the 'split' field in BENCHMARKS configuration.\n"
                            f"{'='*70}\n"
                            f"Original error: {str(e)}"
                        )
                    raise
            
            # Handle DatasetDict (when split=None returns a dictionary of splits)
            from datasets import DatasetDict
            if isinstance(ds, DatasetDict):
                # If split is None, try to use a default split from config or use 'test'
                if split is None:
                    # Try common split names in order of preference
                    preferred_splits = ['test', 'validation', 'val', 'train']
                    selected_split = None
                    for preferred in preferred_splits:
                        if preferred in ds:
                            selected_split = preferred
                            break
                    
                    if selected_split is None:
                        # Use the first available split
                        selected_split = list(ds.keys())[0]
                    
                    ds = ds[selected_split]
                else:
                    # Split was specified but we got a DatasetDict - try to use it
                    if split in ds:
                        ds = ds[split]
                    else:
                        raise ValueError(
                            f"\n{'='*70}\n"
                            f"Split '{split}' not found in dataset\n"
                            f"{'='*70}\n"
                            f"Available splits: {list(ds.keys())}\n"
                            f"{'='*70}"
                        )
            
            # Sample if limit is specified
            if limit and len(ds) > limit:
                import random
                indices = random.sample(range(len(ds)), limit)
                ds = ds.select(indices)
            
            # Get first example to check available fields (for diagnostics)
            first_example = None
            available_fields = []
            if len(ds) > 0:
                first_example = ds[0]
                available_fields = list(first_example.keys())
            
            # Extract texts and labels
            for ex in ds:
                # Try to find text field (case-insensitive)
                # Also try common alternatives: prompt, instructions, text, input
                text = None
                context = None
                text_field_candidates = [text_field, "prompt", "instructions", "text", "input"]
                for candidate in text_field_candidates:
                    for key in ex.keys():
                        if key.lower() == candidate.lower():
                            text = ex[key]
                            break
                    if text:
                        break
                
                # Also check for context field (for HarmBench contextual config)
                for key in ex.keys():
                    if key.lower() == "context":
                        context_val = ex[key]
                        if isinstance(context_val, str) and len(context_val.strip()) > 0:
                            context = context_val
                        break
                
                # Combine prompt and context if both exist
                if text and context:
                    text = f"{context}\n\n{text}"
                elif context and not text:
                    # If only context exists, use it
                    text = context
                
                if not text or not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                
                texts.append(text)
                
                # Extract label
                if label_field:
                    # Try to find label field (case-insensitive)
                    raw_label = None
                    for key in ex.keys():
                        if key.lower() == label_field.lower():
                            raw_label = ex[key]
                            break
                    
                    if raw_label is not None:
                        # Map label
                        if isinstance(raw_label, bool):
                            mapped_label = label_mapping.get(raw_label, "unsafe" if not raw_label else "safe")
                        elif isinstance(raw_label, (int, float)):
                            # Handle numeric labels (e.g., 0, 1)
                            mapped_label = label_mapping.get(int(raw_label), label_mapping.get(raw_label, "unsafe" if raw_label else "safe"))
                        elif isinstance(raw_label, str):
                            # Handle string labels - try exact match first, then case-insensitive
                            if raw_label in label_mapping:
                                mapped_label = label_mapping[raw_label]
                            else:
                                # Try case-insensitive match
                                raw_label_lower = raw_label.lower()
                                found = False
                                for key, value in label_mapping.items():
                                    if isinstance(key, str) and key.lower() == raw_label_lower:
                                        mapped_label = value
                                        found = True
                                        break
                                if not found:
                                    # Default: assume "harmful" or similar means unsafe
                                    mapped_label = "unsafe" if "harm" in raw_label_lower else "safe"
                        elif raw_label in label_mapping:
                            mapped_label = label_mapping[raw_label]
                        else:
                            # Default: assume False/0/negative means unsafe
                            mapped_label = "unsafe" if not raw_label else "safe"
                        labels.append(mapped_label)
                    else:
                        # If label field not found, use default if available
                        default_label = config.get("default_label")
                        if default_label:
                            labels.append(default_label)
                        else:
                            # If no default and label field required, skip
                            texts.pop()
                            continue
                else:
                    # No label field specified - use default_label if available
                    default_label = config.get("default_label")
                    if default_label:
                        labels.append(default_label)
                    else:
                        # No label field and no default - skip
                        texts.pop()
                        continue
                
                # Store metadata
                metadata.append({
                    **{k: v for k, v in ex.items() 
                       if k.lower() not in [text_field.lower(), label_field.lower() if label_field else ""]}
                })
        
        # Apply limit if specified (for JailbreakBench or if limit wasn't applied earlier)
        if limit and len(texts) > limit:
            import random
            indices = random.sample(range(len(texts)), limit)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        
        if len(texts) == 0:
            # Provide helpful error message with available fields
            error_msg = f"No valid examples found in {benchmark_name}. Check field names.\n"
            if first_example is not None:
                error_msg += f"\nAvailable fields in dataset: {available_fields}\n"
                error_msg += f"Looking for text field: '{text_field}' (or alternatives: prompt, instructions, text, input)\n"
                if label_field:
                    error_msg += f"Looking for label field: '{label_field}'\n"
                else:
                    error_msg += f"No label field specified (using default_label: {config.get('default_label', 'None')})\n"
                error_msg += f"\nFirst example keys: {list(first_example.keys())}\n"
                # Show sample values
                for key, value in list(first_example.items())[:5]:
                    if isinstance(value, str):
                        preview = value[:50] + "..." if len(value) > 50 else value
                        error_msg += f"  {key}: {preview}\n"
                    else:
                        error_msg += f"  {key}: {value}\n"
            raise ValueError(error_msg)
        
        return texts, labels, metadata
        
    except Exception as e:
        error_msg = str(e)
        # Check if it's an authentication error
        if "gated dataset" in error_msg.lower() or "must be authenticated" in error_msg.lower():
            raise ValueError(
                f"\n{'='*70}\n"
                f"Authentication Error for {benchmark_name}\n"
                f"{'='*70}\n"
                f"This is a gated dataset that requires HuggingFace authentication.\n\n"
                f"Steps to authenticate:\n"
                f"1. Get a token from: https://huggingface.co/settings/tokens\n"
                f"2. Accept the dataset terms at: https://huggingface.co/datasets/{hf_id}\n"
                f"3. Run with token:\n"
                f"   python benchmark/run_benchmark.py --benchmark {benchmark_name} --hf-token YOUR_TOKEN\n"
                f"   Or set environment variable:\n"
                f"   export HF_TOKEN=YOUR_TOKEN  # Linux/Mac\n"
                f"   set HF_TOKEN=YOUR_TOKEN      # Windows\n"
                f"{'='*70}\n"
                f"Original error: {error_msg}"
            )
        else:
            print(f"Error loading {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            raise


def get_evaluate_function(aggregator_type: str = 'weighted'):
    """
    Get the evaluate_text function for the specified aggregator type.
    
    Args:
        aggregator_type: 'base', 'weighted', 'meta', 'granite', or 'shieldgemma'
        
    Returns:
        The evaluate_text function
    """
    if aggregator_type == 'base':
        if not BASE_AGGREGATOR_AVAILABLE:
            raise ValueError("Base aggregator not available")
        return evaluate_text_base
    elif aggregator_type == 'weighted':
        if not WEIGHTED_AGGREGATOR_AVAILABLE:
            raise ValueError("Weighted aggregator not available")
        return evaluate_text_weighted
    elif aggregator_type == 'meta':
        if not META_AGGREGATOR_AVAILABLE:
            raise ValueError("Meta aggregator not available")
        return evaluate_text_meta
    elif aggregator_type == 'granite':
        if not GRANITE_AVAILABLE:
            raise ValueError("Granite Guardian not available. Install dependencies: pip install transformers torch vllm")
        return evaluate_text_granite
    elif aggregator_type == 'shieldgemma':
        if not SHIELDGEMMA_AVAILABLE:
            raise ValueError("ShieldGemma not available. Install dependencies: pip install transformers[accelerate] torch")
        return evaluate_text_shieldgemma
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}. Must be 'base', 'weighted', 'meta', 'granite', or 'shieldgemma'")


def _binary_prediction_from_individual(
    ind: Dict[str, Any],
    aggregator: str,
    threshold: float,
) -> str:
    """Map per-text expert outputs to safe/unsafe (matches evaluate_on_benchmark batched paths)."""
    if aggregator == "weighted":
        from aggregator.weighted_aggregator import aggregate_results as _agg

        result = _agg(ind, threshold=threshold)
        return "unsafe" if not result["is_safe"] else "safe"
    if aggregator == "base":
        from aggregator.base_aggregator import aggregate_results as _agg

        result = _agg(ind, threshold=threshold)
        return "unsafe" if not result["is_safe"] else "safe"
    if aggregator == "meta":
        from meta_classifier.predict import meta_predict_proba
        from policy.triage import TriageConfig, triage_score
        from policy.dynamic_threshold import context_from_dict

        try:
            p_unsafe = meta_predict_proba(ind)
        except Exception:
            p_unsafe = 0.5
        cfg = TriageConfig.from_env(default_threshold=threshold)
        verdict, _ = triage_score(p_unsafe, config=cfg, ctx=context_from_dict(None))
        is_safe = verdict == "safe"
        return "unsafe" if not is_safe else "safe"
    raise ValueError(f"compare mode only supports base, weighted, meta; got {aggregator}")


def _metrics_binary(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label="unsafe",
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def evaluate_compare_aggregators(
    benchmark_name: str,
    limit: Optional[int] = None,
    threshold: float = 0.5,
    verbose: bool = True,
    hf_token: Optional[str] = None,
    config_override: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Run base, weighted, and meta aggregators on the same benchmark examples (one expert pass).

    Per-domain metrics use metadata['domain'] when present; otherwise the benchmark name is used.
    """
    aggs = ["base", "weighted", "meta"]
    if not BASE_AGGREGATOR_AVAILABLE:
        aggs = [a for a in aggs if a != "base"]
    if not WEIGHTED_AGGREGATOR_AVAILABLE:
        aggs = [a for a in aggs if a != "weighted"]
    if not META_AGGREGATOR_AVAILABLE:
        aggs = [a for a in aggs if a != "meta"]
    if not aggs:
        return {"benchmark": benchmark_name, "error": "No aggregators available for compare"}

    if verbose:
        print(f"\n{'='*70}")
        print(f"Compare aggregators on: {benchmark_name}")
        print(f"Aggregators: {', '.join(aggs)}")
        print(f"{BENCHMARKS[benchmark_name].get('note', '')}")
        print(f"{'='*70}")

    try:
        eval_config = BENCHMARKS[benchmark_name].copy()
        if config_override:
            eval_config["config"] = config_override
        texts, ground_truth_labels, metadata = load_benchmark_dataset(
            benchmark_name, limit, hf_token, config_override=eval_config if config_override else None
        )
        if len(texts) == 0:
            return {"benchmark": benchmark_name, "error": "No examples loaded from dataset"}

        if verbose:
            print(f"Loaded {len(texts)} examples; running experts once (batched)...")

        batched_individuals = run_all_safeguards_batch(texts, batch_size=batch_size)

        domains: List[str] = []
        for m in metadata:
            if isinstance(m, dict) and m.get("domain"):
                domains.append(str(m["domain"]))
            else:
                domains.append(benchmark_name)

        results_by_agg: Dict[str, Any] = {}
        for agg in tqdm(aggs, disable=not verbose, desc="Aggregators"):
            preds = [_binary_prediction_from_individual(ind, agg, threshold) for ind in batched_individuals]
            overall = _metrics_binary(ground_truth_labels, preds)
            overall["n"] = len(preds)
            by_domain: Dict[str, Any] = {}
            for dom in sorted(set(domains)):
                idx = [i for i, d in enumerate(domains) if d == dom]
                if not idx:
                    continue
                yt = [ground_truth_labels[i] for i in idx]
                yp = [preds[i] for i in idx]
                by_domain[dom] = {**_metrics_binary(yt, yp), "n": len(idx)}
            results_by_agg[agg] = {"overall": overall, "by_domain": by_domain}

        if verbose:
            print(f"\n{'='*70}")
            print("Comparison (overall)")
            print(f"{'Aggregator':<12} {'Acc':>8} {'P':>8} {'R':>8} {'F1':>8}")
            print("-" * 50)
            for agg in aggs:
                o = results_by_agg[agg]["overall"]
                print(
                    f"{agg:<12} {o['accuracy']:>7.2%} {o['precision']:>7.2%} "
                    f"{o['recall']:>7.2%} {o['f1_score']:>7.2%}"
                )
            print(f"\nPer-domain F1 (by aggregator):")
            for dom in sorted(set(domains)):
                parts = [f"{agg}={results_by_agg[agg]['by_domain'].get(dom, {}).get('f1_score', float('nan')):.2%}" for agg in aggs]
                print(f"  {dom}: " + " | ".join(parts))

        return {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "aggregators": aggs,
            "total_examples": len(texts),
            "results": results_by_agg,
        }
    except Exception as e:
        if verbose:
            print(f"⚠️  Error in compare mode: {e}")
        import traceback

        return {"benchmark": benchmark_name, "error": str(e), "traceback": traceback.format_exc()}


def evaluate_on_benchmark(
    benchmark_name: str,
    limit: Optional[int] = None,
    threshold: float = 0.5,
    verbose: bool = True,
    hf_token: Optional[str] = None,
    config_override: Optional[str] = None,
    aggregator: str = 'weighted'
) -> Dict[str, Any]:
    """
    Evaluate the safeguarding system on a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark
        limit: Maximum number of examples to evaluate
        threshold: Confidence threshold for flagging
        verbose: Whether to print progress
        aggregator: Aggregator type to use ('base', 'weighted', 'meta', 'granite', or 'shieldgemma', default: 'weighted')
        
    Returns:
        Dictionary with evaluation results
    """
    # Get the appropriate evaluate function
    try:
        evaluate_func = get_evaluate_function(aggregator)
    except ValueError as e:
        return {
            "benchmark": benchmark_name,
            "error": str(e)
        }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {benchmark_name}")
        print(f"Aggregator: {aggregator}")
        print(f"{BENCHMARKS[benchmark_name].get('note', '')}")
        print(f"{'='*70}")
    
    try:
        # Load dataset
        if verbose:
            print("Loading dataset...")
        
        # Create a copy of config for this evaluation (to allow override without mutating global)
        eval_config = BENCHMARKS[benchmark_name].copy()
        if config_override:
            eval_config["config"] = config_override
            if verbose:
                print(f"Using config: {config_override} (override)")
        
        texts, ground_truth_labels, metadata = load_benchmark_dataset(
            benchmark_name, limit, hf_token, config_override=eval_config if config_override else None
        )
        
        if len(texts) == 0:
            return {
                "benchmark": benchmark_name,
                "error": "No examples loaded from dataset"
            }
        
        if verbose:
            print(f"Loaded {len(texts)} examples")
            print(f"Running safeguarding system...\n")
        
        # Run predictions
        predictions = []
        confidences = []
        individual_results = []
        verdicts = []

        # Optional batched expert inference for throughput (weighted/meta/base only).
        batch_enable = os.environ.get("AOS_BENCH_BATCH", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        batch_size = int(os.environ.get("AOS_BENCH_BATCH_SIZE", "8") or "8")

        if batch_enable and aggregator in {"weighted", "meta", "base"}:
            # Run experts in batches, then apply aggregator logic per example.
            batched_individuals = run_all_safeguards_batch(texts, batch_size=batch_size)

            if aggregator == "weighted":
                from aggregator.weighted_aggregator import aggregate_results as _agg
                for ind in tqdm(batched_individuals, disable=not verbose, desc=f"Aggregating ({aggregator})"):
                    result = _agg(ind, threshold=threshold)
                    predictions.append("unsafe" if not result["is_safe"] else "safe")
                    verdicts.append(result.get("verdict", "safe" if result.get("is_safe") else "unsafe"))
                    confidences.append(result.get("average_confidence", 0.0))
                    individual_results.append({"flags": result.get("flags", []), "individual_results": result.get("individual_results", {})})
            elif aggregator == "base":
                from aggregator.base_aggregator import aggregate_results as _agg
                for ind in tqdm(batched_individuals, disable=not verbose, desc=f"Aggregating ({aggregator})"):
                    result = _agg(ind, threshold=threshold)
                    predictions.append("unsafe" if not result["is_safe"] else "safe")
                    verdicts.append(result.get("verdict", "safe" if result.get("is_safe") else "unsafe"))
                    confidences.append(result.get("average_confidence", 0.0))
                    individual_results.append({"flags": result.get("flags", []), "individual_results": result.get("individual_results", {})})
            else:  # meta
                from meta_classifier.predict import meta_predict_proba
                from policy.triage import TriageConfig, triage_score
                from policy.dynamic_threshold import context_from_dict

                cfg = TriageConfig.from_env(default_threshold=threshold)
                for ind in tqdm(batched_individuals, disable=not verbose, desc=f"Aggregating ({aggregator})"):
                    try:
                        p_unsafe = meta_predict_proba(ind)
                    except Exception:
                        p_unsafe = 0.5
                    verdict, triage = triage_score(p_unsafe, config=cfg, ctx=context_from_dict(None))
                    is_safe = verdict == "safe"
                    result = {
                        "is_safe": is_safe,
                        "average_confidence": float(p_unsafe),
                        "verdict": verdict,
                        "triage": triage,
                        "flags": ([] if is_safe else [{"safeguard": "meta", "label": verdict, "confidence": float(p_unsafe)}]),
                        "individual_results": ind,
                    }
                    predictions.append("unsafe" if not is_safe else "safe")
                    verdicts.append(verdict)
                    confidences.append(result["average_confidence"])
                    individual_results.append({"flags": result.get("flags", []), "individual_results": result.get("individual_results", {})})
        else:
            # Default per-example path
            for text in tqdm(texts, disable=not verbose, desc=f"Evaluating ({aggregator})"):
                result = evaluate_func(text, threshold=threshold)

                is_flagged = not result['is_safe']
                predictions.append("unsafe" if is_flagged else "safe")
                verdicts.append(result.get("verdict", "safe" if result.get("is_safe") else "unsafe"))

                confidences.append(result.get('average_confidence', 0.0))

                individual_results.append({
                    'flags': result.get('flags', []),
                    'individual_results': result.get('individual_results', {})
                })
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth_labels,
            predictions,
            average='binary',
            pos_label='unsafe',
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth_labels, predictions, labels=['safe', 'unsafe'])
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                ground_truth_labels,
                predictions,
                labels=['safe', 'unsafe'],
                zero_division=0
            )
        
        # Count predictions
        pred_safe = predictions.count("safe")
        pred_unsafe = predictions.count("unsafe")
        true_safe = ground_truth_labels.count("safe")
        true_unsafe = ground_truth_labels.count("unsafe")
        
        results = {
            "benchmark": benchmark_name,
            "aggregator": aggregator,
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "total_examples": len(texts),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "average_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "confusion_matrix": {
                "safe_safe": int(cm[0][0]),
                "safe_unsafe": int(cm[0][1]),
                "unsafe_safe": int(cm[1][0]),
                "unsafe_unsafe": int(cm[1][1])
            },
            "ground_truth_distribution": {
                "safe": true_safe,
                "unsafe": true_unsafe
            },
            "prediction_distribution": {
                "safe": pred_safe,
                "unsafe": pred_unsafe
            },
            "per_class_metrics": {
                "safe": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support_per_class[0])
                },
                "unsafe": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support_per_class[1])
                }
            }
        }

        # Gray-zone triage stats (if aggregators provide verdict)
        needs_review = sum(1 for v in verdicts if v == "needs_review")
        results["triage"] = {
            "needs_review": int(needs_review),
            "review_rate": float(needs_review / len(verdicts)) if verdicts else 0.0,
            "verdict_counts": {
                "safe": int(sum(1 for v in verdicts if v == "safe")),
                "unsafe": int(sum(1 for v in verdicts if v == "unsafe")),
                "needs_review": int(needs_review),
            },
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Results for {benchmark_name}:")
            print(f"{'='*70}")
            print(f"  Total examples: {results['total_examples']}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1-Score: {f1:.2%}")
            print(f"  Average Confidence: {results['average_confidence']:.2%}")
            print(f"\n  Ground Truth Distribution:")
            print(f"    Safe: {true_safe} ({true_safe/len(texts):.1%})")
            print(f"    Unsafe: {true_unsafe} ({true_unsafe/len(texts):.1%})")
            print(f"\n  Prediction Distribution:")
            print(f"    Safe: {pred_safe} ({pred_safe/len(texts):.1%})")
            print(f"    Unsafe: {pred_unsafe} ({pred_unsafe/len(texts):.1%})")
            print(f"\n  Confusion Matrix:")
            print(f"    Safe → Safe: {cm[0][0]}")
            print(f"    Safe → Unsafe: {cm[0][1]} (False Positives)")
            print(f"    Unsafe → Safe: {cm[1][0]} (False Negatives)")
            print(f"    Unsafe → Unsafe: {cm[1][1]}")
            print(f"\n  Per-Class Metrics:")
            print(f"    Safe - Precision: {precision_per_class[0]:.2%}, Recall: {recall_per_class[0]:.2%}, F1: {f1_per_class[0]:.2%}")
            print(f"    Unsafe - Precision: {precision_per_class[1]:.2%}, Recall: {recall_per_class[1]:.2%}, F1: {f1_per_class[1]:.2%}")
            if results.get("triage"):
                print(f"\n  Triage (gray-zone) Stats:")
                print(f"    Needs review: {results['triage']['needs_review']} ({results['triage']['review_rate']:.2%})")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Error evaluating {benchmark_name}: {e}")
        import traceback
        return {
            "benchmark": benchmark_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_all_benchmarks(
    limit: Optional[int] = None,
    threshold: float = 0.5,
    save_results: bool = True,
    hf_token: Optional[str] = None,
    aggregator: str = 'weighted'
) -> List[Dict[str, Any]]:
    """
    Run evaluation on all available benchmarks.
    
    Args:
        limit: Maximum number of examples per benchmark
        threshold: Confidence threshold for flagging
        save_results: Whether to save results to JSON file
        aggregator: Aggregator type to use ('base', 'weighted', 'granite', or 'shieldgemma', default: 'weighted')
        
    Returns:
        List of results dictionaries
    """
    print("="*70)
    print("ARMY OF SAFEGUARDS - BENCHMARK EVALUATION")
    print("="*70)
    print(f"\nEvaluating on {len(BENCHMARKS)} benchmarks")
    print(f"Aggregator: {aggregator}")
    if limit:
        print(f"Limit: {limit} examples per benchmark")
    print(f"Threshold: {threshold}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for benchmark_name in BENCHMARKS.keys():
        result = evaluate_on_benchmark(
            benchmark_name,
            limit=limit,
            threshold=threshold,
            verbose=True,
            hf_token=hf_token,
            aggregator=aggregator
        )
        all_results.append(result)
    
    # Calculate overall statistics
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    print(f"\nSuccessful evaluations: {len(successful)}/{len(BENCHMARKS)}")
    
    if successful:
        avg_accuracy = sum(r['accuracy'] for r in successful) / len(successful)
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        avg_precision = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        
        print(f"\nAverage Metrics Across Benchmarks:")
        print(f"  Accuracy:  {avg_accuracy:.2%}")
        print(f"  Precision: {avg_precision:.2%}")
        print(f"  Recall:    {avg_recall:.2%}")
        print(f"  F1-Score:  {avg_f1:.2%}")
        
        print(f"\nPer-Benchmark Results:")
        print(f"{'Benchmark':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*70)
        for result in successful:
            print(f"{result['benchmark']:<20} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['f1_score']:<12.2%} "
                  f"{result['precision']:<12.2%} "
                  f"{result['recall']:<12.2%}")
    
    if failed:
        print(f"\nFailed evaluations: {len(failed)}")
        for result in failed:
            print(f"  {result['benchmark']}: {result['error']}")
    
    # Save results
    if save_results and successful:
        output_dir = Path(__file__).parent
        output_file = output_dir / f"benchmark_results_{aggregator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "system": "Army of Safeguards (Aggregator)",
                "aggregator": aggregator,
                "threshold": threshold,
                "limit_per_benchmark": limit,
                "results": all_results,
                "summary": {
                    "avg_accuracy": avg_accuracy,
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_f1_score": avg_f1
                } if successful else {}
            }, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the Army of Safeguards system against public benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single benchmark with weighted aggregator (default)
  python benchmark/run_benchmark.py --benchmark HarmBench --limit 100
  
  # Run with base aggregator
  python benchmark/run_benchmark.py --benchmark HarmBench --aggregator base --limit 100
  
  # Run with Granite Guardian model
  python benchmark/run_benchmark.py --benchmark HarmBench --aggregator granite --limit 100
  
  # Run with ShieldGemma-2b model
  python benchmark/run_benchmark.py --benchmark HarmBench --aggregator shieldgemma --limit 100
  
  # Run all benchmarks with weighted aggregator
  python benchmark/run_benchmark.py --all --limit 100
  
  # Custom threshold
  python benchmark/run_benchmark.py --benchmark WildGuardMix --threshold 0.8 --aggregator base
        """
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(BENCHMARKS.keys()),
        help="Name of the benchmark to run"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available benchmarks"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples per benchmark (default: all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for flagging (default: 0.5)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (sets verbose=False)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for accessing gated datasets (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override dataset config (e.g., for HarmBench: 'standard', 'contextual', or 'copyright')"
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=['base', 'weighted', 'meta', 'granite', 'shieldgemma'],
        default='weighted',
        help="Aggregator type to use: 'base' (threshold-based), 'weighted' (weighted sum, default), 'meta' (learned LR meta-classifier), 'granite' (IBM Granite Guardian), or 'shieldgemma' (Google ShieldGemma-2b)"
    )
    parser.add_argument(
        "--compare-aggregators",
        action="store_true",
        help="On a single benchmark, run base vs weighted vs meta on the same examples (one expert pass) and print per-domain F1",
    )
    
    args = parser.parse_args()
    
    # Use token from command line or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not args.benchmark and not args.all:
        parser.error("Must specify either --benchmark or --all")
    
    if args.benchmark and args.all:
        parser.error("Cannot specify both --benchmark and --all")
    
    if args.compare_aggregators and args.all:
        parser.error("Cannot use --compare-aggregators with --all")

    if args.benchmark:
        if args.compare_aggregators:
            result = evaluate_compare_aggregators(
                args.benchmark,
                limit=args.limit,
                threshold=args.threshold,
                verbose=not args.quiet,
                hf_token=hf_token,
                config_override=args.config,
            )
        else:
            # Run single benchmark
            result = evaluate_on_benchmark(
                args.benchmark,
                limit=args.limit,
                threshold=args.threshold,
                verbose=not args.quiet,
                hf_token=hf_token,
                config_override=args.config,
                aggregator=args.aggregator
            )
        
        if not args.no_save and "error" not in result:
            output_dir = Path(__file__).parent
            suffix = "compare" if args.compare_aggregators else args.aggregator
            output_file = output_dir / f"benchmark_{args.benchmark.lower()}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                payload: Dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(),
                    "system": "Army of Safeguards (Aggregator)",
                    "threshold": args.threshold,
                    "result": result,
                }
                if not args.compare_aggregators:
                    payload["aggregator"] = args.aggregator
                else:
                    payload["mode"] = "compare_aggregators"
                json.dump(payload, f, indent=2)
            print(f"\n✅ Results saved to: {output_file}")
    else:
        # Run all benchmarks
        run_all_benchmarks(
            limit=args.limit,
            threshold=args.threshold,
            save_results=not args.no_save,
            hf_token=hf_token,
            aggregator=args.aggregator
        )

