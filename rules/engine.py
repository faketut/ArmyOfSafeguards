from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

RuleType = Literal["keyword", "regex"]
RuleAction = Literal["block", "tag"]


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    risk_name: str
    action: RuleAction
    tag: str
    span: Optional[Tuple[int, int]] = None


class RuleEngine:
    def __init__(self, rules: Sequence[Dict[str, Any]]):
        self._raw_rules = list(rules)
        self._compiled: List[Tuple[Dict[str, Any], Any]] = []
        self._compile()

    @staticmethod
    def from_file(path: str | Path) -> "RuleEngine":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            raise ValueError("rules file must contain a list at key 'rules'")
        return RuleEngine(rules)

    def _compile(self) -> None:
        compiled: List[Tuple[Dict[str, Any], Any]] = []
        for rule in self._raw_rules:
            rtype: RuleType = rule.get("type")
            pattern = rule.get("pattern")
            if rtype not in ("keyword", "regex"):
                continue
            if not isinstance(pattern, str) or not pattern:
                continue

            flags = re.IGNORECASE if rule.get("case_insensitive") else 0
            if rtype == "keyword":
                compiled.append((rule, (pattern.lower() if flags else pattern, flags)))
            else:
                compiled.append((rule, re.compile(pattern, flags=flags)))
        self._compiled = compiled

    def match(self, text: str) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        if not isinstance(text, str) or not text:
            return matches

        for rule, compiled in self._compiled:
            action: RuleAction = rule.get("action", "tag")
            if action not in ("block", "tag"):
                continue

            risk_name = str(rule.get("risk_name") or "rule")
            tag = str(rule.get("tag") or rule.get("id") or "rule_match")
            rule_id = str(rule.get("id") or tag)

            if rule.get("type") == "keyword":
                needle, flags = compiled
                hay = text.lower() if flags else text
                idx = hay.find(needle)
                if idx != -1:
                    matches.append(
                        RuleMatch(
                            rule_id=rule_id,
                            risk_name=risk_name,
                            action=action,
                            tag=tag,
                            span=(idx, idx + len(needle)),
                        )
                    )
            else:
                m = compiled.search(text)
                if m:
                    matches.append(
                        RuleMatch(
                            rule_id=rule_id,
                            risk_name=risk_name,
                            action=action,
                            tag=tag,
                            span=(m.start(), m.end()),
                        )
                    )
        return matches


def get_default_rule_engine() -> RuleEngine:
    """
    Load default rules shipped with the repo.

    Enabled by setting `AOS_RULES_PATH` or by importing and calling directly.
    """
    base = Path(__file__).parent
    return RuleEngine.from_file(base / "default_rules.json")


def rules_enabled() -> bool:
    return os.environ.get("AOS_ENABLE_RULES", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def load_rule_engine_from_env() -> RuleEngine:
    path = os.environ.get("AOS_RULES_PATH", "").strip()
    if path:
        return RuleEngine.from_file(path)
    return get_default_rule_engine()

