from __future__ import annotations

import re
from pathlib import Path

import yaml


class ReasoningSanitizer:
    def __init__(self, rules_path: Path | str = Path("config/reasoning_rules.yaml")) -> None:
        self.rules_path = Path(rules_path)
        self.universal_patterns: list[str] = []
        self.model_rules: dict[str, dict[str, list[str]]] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        if not self.rules_path.exists():
            self.universal_patterns = []
            self.model_rules = {}
            return

        with self.rules_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        universal = payload.get("universal", {}) if isinstance(payload, dict) else {}
        self.universal_patterns = list(universal.get("patterns", []))
        models = payload.get("models", {}) if isinstance(payload, dict) else {}
        self.model_rules = models if isinstance(models, dict) else {}

    def sanitize(self, text: str, model_name: str, enabled: bool) -> str:
        if not enabled:
            return text

        rules = self.model_rules.get(model_name, {})
        excluded = set(rules.get("exclude_universal", []))
        model_patterns = list(rules.get("patterns", []))

        active_patterns = [p for p in self.universal_patterns if p not in excluded]
        active_patterns.extend(model_patterns)

        cleaned = text
        for pattern in active_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        if cleaned.strip() == "":
            return text

        return cleaned.strip()
