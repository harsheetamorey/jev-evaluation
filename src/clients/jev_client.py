"""Thin wrappers around TypeSafe's Jev clients, plus the AsyncEvaluator adapter."""

import time
from types import TracebackType
from typing import Protocol, Self

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    JSONContent,
    Questions,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)

from config import settings
from models.prediction import Example, PredictionResult


class JevClient:
    """Constructs a `TypeSafeClient` from project settings and exposes `system_one`."""

    def __init__(self) -> None:
        self._client = TypeSafeClient(api_key=settings.typesafe_api_key, model=settings.jev_model)

    def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse:
        return self._client.system_one(state, questions)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncJevClient:
    """Async counterpart of `JevClient`, for concurrent Jev calls."""

    def __init__(self) -> None:
        self._client = AsyncTypeSafeClient(api_key=settings.typesafe_api_key, model=settings.jev_model)

    async def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse:
        return await self._client.system_one(state, questions)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SystemOneCaller(Protocol):
    """What `JevEvaluator` needs from a client: an async `system_one`. Satisfied by `AsyncJevClient`."""

    async def system_one(self, state: JSONContent, questions: Questions) -> SystemOneResponse: ...


class JevEvaluator:
    """Adapts a Jev client to `evaluation.runner.AsyncEvaluator`: predict(example) -> PredictionResult.

    Answers a single Choice question per example, built from that example's own
    candidates. Never retries beyond the SDK's own retry policy: an SDK error is
    caught once and recorded on the result, not retried here.
    """

    provider = "jev"

    def __init__(self, client: SystemOneCaller, experiment: str) -> None:
        self._client = client
        self._experiment = experiment

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
            )

        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices["intent"]
        prediction = answer.choice
        correct = prediction == example.ground_truth if example.ground_truth is not None else None
        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None

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
        )
