from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, List


PredictFn = Callable[[str], Dict[str, Any]]
PredictBatchFn = Callable[[Sequence[str]], List[Dict[str, Any]]]


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    predict: PredictFn
    predict_batch: Optional[PredictBatchFn] = None


def get_expert_registry() -> Mapping[str, ExpertSpec]:
    """
    Single source of truth for which experts exist.

    Add new experts here (and keep the `name` stable for downstream metrics).
    """
    # Avoid importing experts (which may download/load HF models) just to list names.
    # We store module paths and import predict lazily at call-time.
    from importlib import import_module

    module_by_name = {
        "factuality": "experts.factuality",
        "toxicity": "experts.toxicity",
        "sexual": "experts.sexual",
        "jailbreak": "experts.jailbreak",
    }

    def _lazy_predict(module_path: str) -> PredictFn:
        def _predict(text: str) -> Dict[str, Any]:
            mod = import_module(module_path)
            return mod.predict(text)

        return _predict

    def _lazy_predict_batch(module_path: str) -> PredictBatchFn:
        def _predict_batch(texts: Sequence[str]) -> List[Dict[str, Any]]:
            mod = import_module(module_path)
            if not hasattr(mod, "predict_batch"):
                raise AttributeError(f"{module_path} does not implement predict_batch")
            return mod.predict_batch(texts)

        return _predict_batch

    return {
        name: ExpertSpec(
            name=name,
            predict=_lazy_predict(mod),
            predict_batch=_lazy_predict_batch(mod),
        )
        for name, mod in module_by_name.items()
    }


def run_expert(name: str, text: str) -> Dict[str, Any]:
    reg = get_expert_registry()
    if name not in reg:
        raise KeyError(f"Unknown expert: {name}. Available: {sorted(reg.keys())}")
    return reg[name].predict(text)

