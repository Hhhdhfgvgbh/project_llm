from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BlockRule:
    start: str
    end: str


@dataclass(frozen=True)
class CleaningRuleSet:
    remove_blocks: list[BlockRule]
    disabled_universal_blocks: list[int]


class OutputCleaner:
    def __init__(self, universal_rules: list[BlockRule], model_rules: dict[str, CleaningRuleSet]) -> None:
        self.universal_rules = universal_rules
        self.model_rules = model_rules

    @classmethod
    def from_yaml(cls, path: Path | str) -> "OutputCleaner":
        source = Path(path)
        if not source.exists():
            return cls(universal_rules=[], model_rules={})

        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        universal = cls._parse_blocks(payload.get("universal", {}).get("remove_blocks", []))
        model_rules: dict[str, CleaningRuleSet] = {}

        raw_models = payload.get("model_rules", {}) or {}
        for model_name, config in raw_models.items():
            config = config or {}
            model_rules[model_name] = CleaningRuleSet(
                remove_blocks=cls._parse_blocks(config.get("remove_blocks", [])),
                disabled_universal_blocks=[int(idx) for idx in config.get("disabled_universal_blocks", [])],
            )

        return cls(universal_rules=universal, model_rules=model_rules)

    def clean(self, text: str, model_name: str) -> str:
        rules = list(self.universal_rules)
        model_rule = self.model_rules.get(model_name)
        if model_rule:
            rules = [rule for idx, rule in enumerate(rules) if idx not in set(model_rule.disabled_universal_blocks)]
            rules.extend(model_rule.remove_blocks)

        cleaned = text
        for rule in rules:
            pattern = re.compile(re.escape(rule.start) + r".*?" + re.escape(rule.end), re.IGNORECASE | re.DOTALL)
            cleaned = pattern.sub("", cleaned)

        # cleanup common leftovers
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        return cleaned or text

    @staticmethod
    def _parse_blocks(items: list[dict]) -> list[BlockRule]:
        blocks: list[BlockRule] = []
        for item in items:
            start = str(item.get("start", "")).strip()
            end = str(item.get("end", "")).strip()
            if not start or not end:
                continue
            blocks.append(BlockRule(start=start, end=end))
        return blocks
