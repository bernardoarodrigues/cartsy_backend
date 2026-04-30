from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "text-embedding-3-small": {"input": 0.02},
}


def usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def usage_nested_value(usage: Any, parent_name: str, child_name: str) -> int:
    parent = usage.get(parent_name) if isinstance(usage, dict) else getattr(usage, parent_name, None)
    if parent is None:
        return 0
    value = parent.get(child_name) if isinstance(parent, dict) else getattr(parent, child_name, None)
    return int(value or 0)


@dataclass
class StageMetric:
    elapsed_seconds: float = 0.0
    items: int = 0

    def as_report(self) -> dict[str, float | int | None]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "items": self.items,
            "avg_seconds_per_item": round(self.elapsed_seconds / self.items, 6) if self.items else None,
        }


@dataclass
class UsageAccumulator:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.calls += 1
        input_tokens = usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = usage_value(usage, "total_tokens")
        cached_tokens = usage_nested_value(usage, "input_tokens_details", "cached_tokens")
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens or input_tokens + output_tokens

    def cost_usd(self, model: str) -> float:
        prices = USD_PER_1M_TOKENS.get(model, {})
        billable_input = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            billable_input * prices.get("input", 0.0)
            + self.cached_input_tokens * prices.get("cached_input", prices.get("input", 0.0))
            + self.output_tokens * prices.get("output", 0.0)
        ) / 1_000_000

    def as_report(self, model: str) -> dict[str, float | int | str | None]:
        return {
            "model": model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_usd(model), 6),
            "pricing_note": "Estimated with standard OpenAI prices per 1M tokens configured in pipeline metrics.",
        }


@dataclass
class RunMetrics:
    stages: dict[str, StageMetric] = field(default_factory=dict)
    openai_usage: dict[str, UsageAccumulator] = field(default_factory=lambda: defaultdict(UsageAccumulator))

    @contextmanager
    def stage(self, name: str, *, items: int = 0):
        started = perf_counter()
        try:
            yield
        finally:
            metric = self.stages.setdefault(name, StageMetric())
            metric.elapsed_seconds += perf_counter() - started
            metric.items += items

    def add_usage(self, model: str, usage: Any) -> None:
        self.openai_usage[model].add(usage)

    def as_report(
        self,
        *,
        embedding_model: str,
        extraction_model: str,
        input_records: int,
        total_elapsed_seconds: float,
    ) -> dict[str, object]:
        usage_by_model = {
            model: usage.as_report(model)
            for model, usage in sorted(self.openai_usage.items())
        }
        total_cost = sum(usage.cost_usd(model) for model, usage in self.openai_usage.items())
        return {
            "timing": {
                "total_elapsed_seconds": round(total_elapsed_seconds, 3),
                "input_records": input_records,
                "avg_seconds_per_input_record": round(total_elapsed_seconds / input_records, 6) if input_records else None,
                "stages": {name: metric.as_report() for name, metric in self.stages.items()},
            },
            "openai": {
                "embedding_model": embedding_model,
                "extraction_model": extraction_model,
                "usage_by_model": usage_by_model,
                "total_estimated_cost_usd": round(total_cost, 6),
                "cost_source": "OpenAI standard pricing checked 2026-04-30; update USD_PER_1M_TOKENS if model prices change.",
            },
        }

