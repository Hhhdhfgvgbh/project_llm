from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from app.config.schemas import AggregationConfig, AggregationType


def normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"```.*?```", "", normalized, flags=re.DOTALL)
    normalized = re.sub(r"`([^`]*)`", r"\1", normalized)
    normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(".")
    return normalized


@dataclass
class AggregationResult:
    output: str
    details: dict


class AggregationEngine:
    def __init__(self, synthesis_callback: Callable[[str, str], str] | None = None) -> None:
        self.synthesis_callback = synthesis_callback

    def aggregate(self, responses: list[str], config: AggregationConfig) -> AggregationResult:
        if not responses:
            raise ValueError("No responses provided for aggregation")

        if config.type == AggregationType.CONCAT:
            return AggregationResult(output="\n\n".join(responses), details={"strategy": "concat"})

        if config.type == AggregationType.MAJORITY_VOTE:
            return self._majority_vote(responses)

        if config.type == AggregationType.SYNTHESIS:
            if not self.synthesis_callback or not config.synthesis_model:
                raise ValueError("Synthesis callback and synthesis_model are required")
            prompt = self._build_synthesis_prompt(responses)
            return AggregationResult(
                output=self.synthesis_callback(config.synthesis_model, prompt),
                details={"strategy": "synthesis", "synthesis_model": config.synthesis_model, "prompt": prompt},
            )

        if config.type == AggregationType.CUSTOM_TEMPLATE:
            if not self.synthesis_callback or not config.synthesis_model or not config.template:
                raise ValueError("Custom template requires synthesis callback, synthesis_model and template")
            rendered = config.template
            for index, response in enumerate(responses, start=1):
                rendered = rendered.replace(f"{{{{model_{index}}}}}", response)
            return AggregationResult(
                output=self.synthesis_callback(config.synthesis_model, rendered),
                details={
                    "strategy": "custom_template",
                    "synthesis_model": config.synthesis_model,
                    "template": config.template,
                    "rendered_prompt": rendered,
                },
            )

        raise ValueError(f"Unknown aggregation type: {config.type}")

    def _majority_vote(self, responses: list[str]) -> AggregationResult:
        normalized = [normalize_text(item) for item in responses]
        groups: dict[int, list[int]] = defaultdict(list)
        assigned: list[int] = []

        for index, response in enumerate(normalized):
            if index in assigned:
                continue

            groups[index].append(index)
            assigned.append(index)

            for other_index in range(index + 1, len(normalized)):
                if other_index in assigned:
                    continue
                score = SequenceMatcher(None, response, normalized[other_index]).ratio()
                threshold = 0.85 if max(len(response), len(normalized[other_index])) < 200 else 0.9
                if score >= threshold:
                    groups[index].append(other_index)
                    assigned.append(other_index)

        winner = max(
            groups.values(),
            key=lambda member_ids: (len(member_ids), len(responses[member_ids[0]]), -member_ids[0]),
        )
        winner_idx = winner[0]
        return AggregationResult(
            output=responses[winner_idx],
            details={
                "strategy": "majority_vote",
                "groups": {str(k): v for k, v in groups.items()},
                "winner_group": winner,
                "winner_index": winner_idx,
            },
        )

    @staticmethod
    def _build_synthesis_prompt(responses: list[str]) -> str:
        parts = ["Synthesize the best final answer from candidate responses:\n"]
        for index, response in enumerate(responses, start=1):
            parts.append(f"Response {index}:\n{response}\n")
        return "\n".join(parts)
