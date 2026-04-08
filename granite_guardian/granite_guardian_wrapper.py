"""
Granite Guardian Wrapper for Army of Safeguards.

This module provides a wrapper around IBM Granite Guardian 3.3 8B model
that is compatible with the benchmark evaluation framework.

Granite Guardian is a specialized model for judging if prompts and responses
meet specified safety criteria (jailbreak, profanity, hallucinations, etc.).
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
import warnings
from wrappers.logging_utils import get_logger

# Try to import required dependencies
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch not available. Install with: pip install torch")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Fallback to transformers pipeline if vLLM not available
    try:
        from transformers import pipeline
        PIPELINE_AVAILABLE = True
    except ImportError:
        PIPELINE_AVAILABLE = False

# Model configuration
MODEL_NAME = "ibm-granite/granite-guardian-3.3-8b"
SAFE_TOKEN = "no"  # "no" means safe
RISKY_TOKEN = "yes"  # "yes" means unsafe

# Global model and tokenizer instances (lazy loading)
_model = None
_tokenizer = None
_pipeline = None
_log = get_logger("granite_guardian_wrapper")
_load_error: Optional[str] = None


def _parse_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse Granite Guardian response to extract score and reasoning trace.
    
    Args:
        response: Raw model output
        
    Returns:
        Tuple of (score, trace) where score is 'yes' or 'no', trace is reasoning (if available)
    """
    # Granite Guardian uses <think> or <think> tags for reasoning
    # and <score> tags for the final judgment
    trace_match = re.findall(r'<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>', response, re.DOTALL)
    score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)
    
    score, trace = None, None
    
    if trace_match:
        trace = trace_match[-1].strip()
    
    if score_match:
        score = score_match[-1].strip().lower()
    
    return score, trace


def _load_model_vllm(device: str = "cuda"):
    """Load model using vLLM (faster inference)."""
    global _model, _tokenizer
    global _load_error

    if _load_error is not None:
        raise RuntimeError(_load_error)
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    if not VLLM_AVAILABLE:
        _load_error = "vLLM not available. Install with: pip install vllm"
        raise ImportError(_load_error)
    
    # Load tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise
    
    # Load model with vLLM
    try:
        _model = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            trust_remote_code=True
        )
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise
    
    return _model, _tokenizer


def _load_model_pipeline(device: str = "cuda"):
    """Load model using transformers pipeline (fallback if vLLM not available)."""
    global _pipeline, _tokenizer
    global _load_error

    if _load_error is not None:
        raise RuntimeError(_load_error)
    
    if _pipeline is not None and _tokenizer is not None:
        return _pipeline, _tokenizer
    
    if not PIPELINE_AVAILABLE:
        _load_error = "transformers pipeline not available. Install with: pip install transformers"
        raise ImportError(_load_error)
    
    # Load tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise
    
    # Load model with pipeline
    device_map = "auto" if device == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        _pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            tokenizer=_tokenizer,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise
    
    return _pipeline, _tokenizer


def _load_model(device: str = "cuda", prefer_vllm: bool = True):
    """
    Load the Granite Guardian model.
    
    Args:
        device: Device to use ("cuda" or "cpu")
        prefer_vllm: Whether to prefer vLLM over transformers pipeline
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if prefer_vllm and VLLM_AVAILABLE:
        try:
            return _load_model_vllm(device)
        except Exception as e:
            _log.warning("Failed to load with vLLM (%s). Falling back to transformers pipeline.", e)
            return _load_model_pipeline(device)
    else:
        return _load_model_pipeline(device)


def _generate_with_vllm(model, tokenizer, chat: str, think: bool = False) -> str:
    """Generate response using vLLM."""
    nlogprobs = 20
    max_tokens = 2048
    
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=nlogprobs,
        max_tokens=max_tokens
    )
    
    output = model.generate(chat, sampling_params, use_tqdm=False)
    return output[0].outputs[0].text.strip()


def _generate_with_pipeline(pipeline, tokenizer, chat: str, think: bool = False) -> str:
    """Generate response using transformers pipeline."""
    max_tokens = 2048
    
    outputs = pipeline(
        chat,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        return_full_text=False
    )
    
    # Extract generated text
    if isinstance(outputs, list) and len(outputs) > 0:
        response = outputs[0].get('generated_text', '')
    else:
        response = str(outputs)
    
    return response.strip()


def _score_unsafe_from_score_token(score: Optional[str]) -> float:
    # "yes" = unsafe, "no" = safe
    if score == RISKY_TOKEN:
        return 1.0
    if score == SAFE_TOKEN:
        return 0.0
    return 0.5


def evaluate_text(
    text: str,
    threshold: float = 0.5,
    criteria: str = "jailbreak",
    think_mode: bool = False,
    device: str = "cuda",
    documents: Optional[list] = None,
    tools: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluate text using Granite Guardian model.
    
    This function is compatible with the benchmark framework and follows
    the same interface as the aggregator evaluate_text functions.
    
    Args:
        text: Input text to evaluate
        threshold: Confidence threshold for flagging (not used directly, but kept for compatibility)
        criteria: Safety criterion to check. Options:
            - "jailbreak": Detect jailbreak attempts
            - "profanity": Detect profane language
            - "function_call": Detect tool-call hallucinations (requires tools parameter)
            - "groundedness": Check if response is grounded in context (requires documents parameter)
        think_mode: Enable thinking mode for detailed reasoning traces
        device: Device to run on ("cuda" or "cpu")
        documents: List of documents for groundedness checking (format: [{'doc_id': str, 'text': str}])
        tools: List of tools for function_call checking (format: [{'name': str, 'description': str, ...}])
        
    Returns:
        Dictionary containing:
            - is_safe: Boolean indicating if content is safe
            - average_confidence: Confidence score (1.0 if unsafe, 0.0 if safe)
            - flags: List of flags (empty if safe, contains one flag if unsafe)
            - individual_results: Dictionary with detailed results
            - score: Raw score from model ("yes" or "no")
            - trace: Reasoning trace (if think_mode enabled)
    """
    from wrappers.utils import apply_triage, get_effective_device, safe_error_result

    effective_device = get_effective_device(device)

    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        return safe_error_result(
            wrapper_name="granite_guardian",
            reason="transformers not available",
            threshold=threshold,
        )
    
    # Validate criteria
    valid_criteria = ["jailbreak", "profanity", "function_call", "groundedness"]
    if criteria not in valid_criteria:
        warnings.warn(f"Unknown criteria '{criteria}'. Using 'jailbreak' instead.")
        criteria = "jailbreak"
    
    # Validate criteria-specific requirements
    if criteria == "function_call" and tools is None:
        warnings.warn("function_call criteria requires tools parameter. Using empty list.")
        tools = []
    
    if criteria == "groundedness" and documents is None:
        warnings.warn("groundedness criteria requires documents parameter. Using empty list.")
        documents = []
    
    try:
        # Load model (lazy loading)
        model, tokenizer = _load_model(device=effective_device, prefer_vllm=True)
        
        # Prepare messages
        messages = [{"role": "user", "content": text}]
        
        # For groundedness, we need assistant response
        if criteria == "groundedness":
            # If text contains both prompt and response, split them
            # Otherwise, assume text is the response
            messages = [{"role": "assistant", "content": text}]
        
        # Prepare guardian config
        guardian_config = {"criteria_id": criteria}
        
        # Prepare chat template
        chat_kwargs = {
            "guardian_config": guardian_config,
            "think": think_mode,
            "tokenize": False,
            "add_generation_prompt": True
        }
        
        # Add optional parameters
        if documents is not None:
            chat_kwargs["documents"] = documents
        if tools is not None:
            chat_kwargs["available_tools"] = tools
        
        # Apply chat template
        chat = tokenizer.apply_chat_template(messages, **chat_kwargs)
        
        # Generate response
        # Check if model is vLLM LLM instance (check by type name to avoid import issues)
        is_vllm = VLLM_AVAILABLE and hasattr(model, 'llm_engine')
        
        if is_vllm:
            response_text = _generate_with_vllm(model, tokenizer, chat, think_mode)
        else:
            response_text = _generate_with_pipeline(model, tokenizer, chat, think_mode)
        
        # Parse response
        score, trace = _parse_response(response_text)

        score_unsafe = _score_unsafe_from_score_token(score)
        payload = {
            "criteria": criteria,
            "score": score,
            "trace": trace if think_mode else None,
            "device": effective_device,
            "model": MODEL_NAME,
            "score_unsafe": float(score_unsafe),
        }

        out = apply_triage(
            wrapper_name="granite_guardian",
            score_unsafe=float(score_unsafe),
            threshold=threshold,
            individual_payload=payload,
        )
        out["score"] = score
        out["trace"] = trace
        return out
        
    except Exception as e:
        _log.warning("Granite Guardian evaluate_text failed: %s", e)
        return safe_error_result(
            wrapper_name="granite_guardian",
            reason=f"{type(e).__name__}: {e}",
            threshold=threshold,
        )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = input("Enter text to evaluate: ")
    
    print("\nRunning Granite Guardian...")
    print("=" * 60)
    
    result = evaluate_text(test_text, criteria="jailbreak")
    
    print(f"\nOverall Safety: {'✅ SAFE' if result['is_safe'] else '⚠️  FLAGGED'}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Confidence: {result['average_confidence']:.2%}")
    
    if result['flags']:
        print(f"\nFlags ({len(result['flags'])}):")
        for flag in result['flags']:
            print(f"  - {flag['safeguard']} ({flag['criteria']}): {flag['score']}")
    
    if result.get('trace'):
        print(f"\nReasoning Trace:")
        print(result['trace'][:500] + "..." if len(result['trace']) > 500 else result['trace'])
    
    print("=" * 60)

