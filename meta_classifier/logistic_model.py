from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class LogisticArtifact:
    feature_names: List[str]
    coef: List[float]
    intercept: float
    # Post-hoc temperature on probability: p' = sigmoid(logit(p) / T). 1.0 = disabled.
    temperature: float = 1.0

    @staticmethod
    def load(path: str | Path) -> "LogisticArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return LogisticArtifact(
            feature_names=list(payload["feature_names"]),
            coef=[float(x) for x in payload["coef"]],
            intercept=float(payload["intercept"]),
            temperature=float(payload.get("temperature", 1.0)),
        )

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "coef": self.coef,
                    "intercept": self.intercept,
                    "temperature": self.temperature,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _sigmoid(z: float) -> float:
    # stable enough for small LR models
    import math

    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float, eps: float = 1e-9) -> float:
    import math

    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def predict_proba(artifact: LogisticArtifact, x: Sequence[float]) -> float:
    if len(x) != len(artifact.coef):
        raise ValueError(f"feature length mismatch: got {len(x)} expected {len(artifact.coef)}")
    z = artifact.intercept
    for w, xi in zip(artifact.coef, x):
        z += w * float(xi)
    p = _sigmoid(z)
    t = float(artifact.temperature)
    if t <= 0:
        t = 1.0
    if abs(t - 1.0) < 1e-12:
        return p
    return _sigmoid(_logit(p) / t)

