"""Shared data models: what an evaluator receives and what it must return."""

from dataclasses import dataclass


@dataclass
class Example:
    """A single labeled item to be classified, dataset-agnostic."""

    example_id: str
    dataset: str
    state: str
    candidates: list[str]
    ground_truth: str | None = None
    locale: str | None = None


@dataclass
class PredictionResult:
    """The common output shape every evaluator (rules, Jev, LLM) must produce."""

    experiment: str
    provider: str

    example_id: str
    dataset: str
    locale: str | None

    ground_truth: str | None
    prediction: str | None
    correct: bool | None

    confidence: float | None

    latency_ms: float

    candidates: list[str]

    error: str | None
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    text: str | None = None
