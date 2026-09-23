"""Unit tests for the generic Evaluator protocol and run_evaluation driver."""

from dataclasses import dataclass, field

from evaluation.runner import Evaluator, run_evaluation
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
