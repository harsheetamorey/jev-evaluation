"""Unit tests for the generic Evaluator protocols and run_evaluation drivers."""

import asyncio
from dataclasses import dataclass, field

from evaluation.runner import AsyncEvaluator, Evaluator, run_evaluation, run_evaluation_async
from models.prediction import Example, PredictionResult


@dataclass
class FakeEvaluator:
    """A trivial Evaluator that always predicts the first candidate."""

    provider: str = "fake"
    calls: list[Example] = field(default_factory=list)

    def predict(self, example: Example) -> PredictionResult:
        self.calls.append(example)
        prediction = example.candidates[0]
        return PredictionResult(
            experiment="unit-test",
            provider=self.provider,
            example_id=example.example_id,
            dataset=example.dataset,
            locale=example.locale,
            ground_truth=example.ground_truth,
            prediction=prediction,
            correct=prediction == example.ground_truth,
            confidence=1.0,
            latency_ms=0.0,
            candidates=example.candidates,
            error=None,
        )


def test_fake_evaluator_satisfies_evaluator_protocol() -> None:
    assert isinstance(FakeEvaluator(), Evaluator)


def test_run_evaluation_calls_predict_once_per_example() -> None:
    evaluator = FakeEvaluator()
    examples = [
        Example(example_id="ex-1", dataset="bitext", state="a", candidates=["x", "y"], ground_truth="x"),
        Example(example_id="ex-2", dataset="bitext", state="b", candidates=["y", "x"], ground_truth="z"),
    ]

    results = run_evaluation(evaluator, examples)

    assert [r.example_id for r in results] == ["ex-1", "ex-2"]
    assert results[0].correct is True
    assert results[1].correct is False
    assert len(evaluator.calls) == 2


@dataclass
class FakeAsyncEvaluator:
    """An async Evaluator that tracks in-flight call count to verify concurrency bounds."""

    provider: str = "fake-async"
    delay_seconds: float = 0.02
    in_flight: int = field(default=0)
    max_in_flight: int = field(default=0)

    async def predict(self, example: Example) -> PredictionResult:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay_seconds)
        self.in_flight -= 1
        prediction = example.candidates[0]
        return PredictionResult(
            experiment="unit-test",
            provider=self.provider,
            example_id=example.example_id,
            dataset=example.dataset,
            locale=example.locale,
            ground_truth=example.ground_truth,
            prediction=prediction,
            correct=prediction == example.ground_truth,
            confidence=1.0,
            latency_ms=0.0,
            candidates=example.candidates,
            error=None,
        )


def test_fake_async_evaluator_satisfies_async_evaluator_protocol() -> None:
    assert isinstance(FakeAsyncEvaluator(), AsyncEvaluator)


def test_run_evaluation_async_preserves_order() -> None:
    evaluator = FakeAsyncEvaluator()
    examples = [
        Example(example_id=f"ex-{i}", dataset="bitext", state=str(i), candidates=["x", "y"], ground_truth="x")
        for i in range(6)
    ]

    results = asyncio.run(run_evaluation_async(evaluator, examples, concurrency=3))

    assert [r.example_id for r in results] == [f"ex-{i}" for i in range(6)]


def test_run_evaluation_async_respects_concurrency_bound() -> None:
    evaluator = FakeAsyncEvaluator(delay_seconds=0.02)
    examples = [
        Example(example_id=f"ex-{i}", dataset="bitext", state=str(i), candidates=["x"], ground_truth="x")
        for i in range(10)
    ]

    asyncio.run(run_evaluation_async(evaluator, examples, concurrency=3))

    assert evaluator.max_in_flight <= 3
