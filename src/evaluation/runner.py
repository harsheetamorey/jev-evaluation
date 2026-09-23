"""Generic evaluation runner: drives any Evaluator over any iterable of Examples."""

import asyncio
import time
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from typesafe_sdk import Choice, JSONContent, Questions, SystemOneResponse, TypeSafeError

from evaluation.pricing import estimate_cost_usd
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


class SystemOneCaller(Protocol):
    """What `SystemOneEvaluator` needs from a client: an async `system_one`.

    Satisfied by both `clients.jev_client.AsyncJevClient` (Jev) and
    `clients.llm_client.AsyncLlmClient` (an LLM via TypeSafe's System One
    Adapter) -- the adapter's response subclasses `typesafe_sdk.SystemOneResponse`,
    so one evaluator implementation works for both.
    """

    async def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse: ...


class SystemOneEvaluator:
    """Adapts any System-One-compatible client to AsyncEvaluator: predict(example) -> PredictionResult.

    Answers a single Choice question per example, built from that example's own
    candidates. Never retries beyond the client's own retry policy: an error is
    caught once and recorded on the result, not retried here. `provider` should
    be an explicit, specific label (e.g. "jev", or an LLM's exact model name
    like "gpt-4o-mini") -- never a generic "llm".
    """

    def __init__(self, client: SystemOneCaller, experiment: str, provider: str) -> None:
        self._client = client
        self._experiment = experiment
        self.provider = provider

    async def predict(self, example: Example) -> PredictionResult:
        start = time.perf_counter()
        try:
            response = await self._client.system_one(
                state={"user_message": example.state},
                questions={
                    "intent": Choice(
                        instructions="Select the intent that best represents the user's request.",
                        criteria={choice: None for choice in example.candidates},
                    )
                },
            )
        except TypeSafeError as exc:
            return PredictionResult(
                experiment=self._experiment,
                provider=self.provider,
                example_id=example.example_id,
                dataset=example.dataset,
                locale=example.locale,
                ground_truth=example.ground_truth,
                prediction=None,
                correct=None,
                confidence=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                candidates=example.candidates,
                error=str(exc),
                request_id=getattr(exc, "request_id", None),
                text=example.state,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices["intent"]
        prediction = answer.choice
        correct = prediction == example.ground_truth if example.ground_truth is not None else None
        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None

        usage = response.usage
        input_tokens_total = getattr(usage, "input_tokens_total", None)
        input_tokens = input_tokens_total if input_tokens_total is not None else usage.input_tokens
        output_tokens_total = getattr(usage, "output_tokens_total", None)
        output_tokens = output_tokens_total if output_tokens_total is not None else usage.output_tokens

        return PredictionResult(
            experiment=self._experiment,
            provider=self.provider,
            example_id=example.example_id,
            dataset=example.dataset,
            locale=example.locale,
            ground_truth=example.ground_truth,
            prediction=prediction,
            correct=correct,
            confidence=answer.probabilities.get(prediction),
            latency_ms=latency_ms,
            candidates=example.candidates,
            error=None,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost_usd(response.model, input_tokens, output_tokens),
            text=example.state,
        )


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
