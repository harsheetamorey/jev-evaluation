"""Generic evaluation runner: drives any Evaluator over any iterable of Examples."""

import asyncio
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from models.prediction import Example, PredictionResult


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can turn an Example into a PredictionResult synchronously (e.g. rules)."""

    provider: str

    def predict(self, example: Example) -> PredictionResult: ...


@runtime_checkable
class AsyncEvaluator(Protocol):
    """Anything that can turn an Example into a PredictionResult asynchronously (Jev, LLM)."""

    provider: str

    async def predict(self, example: Example) -> PredictionResult: ...


def run_evaluation(evaluator: Evaluator, examples: Iterable[Example]) -> list[PredictionResult]:
    """Run `evaluator` over every example, returning one PredictionResult per example."""
    return [evaluator.predict(example) for example in examples]


async def run_evaluation_async(
    evaluator: AsyncEvaluator,
    examples: Iterable[Example],
    concurrency: int,
) -> list[PredictionResult]:
    """Run `evaluator` over every example with at most `concurrency` requests in flight.

    Returns results in the same order as `examples`, regardless of completion order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def predict_one(example: Example) -> PredictionResult:
        async with semaphore:
            return await evaluator.predict(example)

    return await asyncio.gather(*(predict_one(example) for example in examples))
