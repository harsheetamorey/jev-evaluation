"""Generic evaluation runner: drives any Evaluator over any iterable of Examples."""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from models.prediction import Example, PredictionResult


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can turn an Example into a PredictionResult (rules, Jev, LLM)."""

    provider: str

    def predict(self, example: Example) -> PredictionResult: ...


def run_evaluation(evaluator: Evaluator, examples: Iterable[Example]) -> list[PredictionResult]:
    """Run `evaluator` over every example, returning one PredictionResult per example."""
    return [evaluator.predict(example) for example in examples]
