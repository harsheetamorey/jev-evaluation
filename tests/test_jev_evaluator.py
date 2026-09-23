"""Unit tests for JevEvaluator's mapping from Jev responses to PredictionResult.

Uses a fake client (no network, no API key) so these run without credentials.
"""

import asyncio

from typesafe_sdk import TypeSafeError

from clients.jev_client import JevEvaluator
from models.prediction import Example


class _FakeAnswer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice = choice
        self.probabilities = probabilities


class _FakeResponse:
    def __init__(self, choice: str, probabilities: dict[str, float], request_id: str | None = "req-abc") -> None:
        self.choices = {"intent": _FakeAnswer(choice, probabilities)}
        self._request_id = request_id

    @property
    def request_id(self) -> str:
        if self._request_id is None:
            raise TypeSafeError("The response did not include a request ID.")
        return self._request_id


class _FakeClient:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    async def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self._error is not None:
            raise self._error
        return self._response


def _example(**overrides) -> Example:
    defaults = dict(
        example_id="ex-1",
        dataset="bitext",
        state="Where is my order?",
        candidates=["track_order", "cancel_order", "other"],
        ground_truth="track_order",
        locale=None,
    )
    return Example(**{**defaults, **overrides})


def test_predict_maps_correct_choice_and_confidence() -> None:
    client = _FakeClient(response=_FakeResponse("track_order", {"track_order": 0.8, "cancel_order": 0.2}))
    evaluator = JevEvaluator(client, experiment="unit-test")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.prediction == "track_order"
    assert result.correct is True
    assert result.confidence == 0.8
    assert result.error is None
    assert result.request_id == "req-abc"
    assert result.candidates == ["track_order", "cancel_order", "other"]


def test_predict_marks_incorrect_when_choice_mismatches_ground_truth() -> None:
    client = _FakeClient(response=_FakeResponse("other", {"other": 0.5}))
    evaluator = JevEvaluator(client, experiment="unit-test")

    result = asyncio.run(evaluator.predict(_example(ground_truth="track_order")))

    assert result.prediction == "other"
    assert result.correct is False


def test_predict_sends_state_and_criteria_built_from_candidates() -> None:
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}))
    evaluator = JevEvaluator(client, experiment="unit-test")

    asyncio.run(evaluator.predict(_example(state="hello world", candidates=["x", "y"])))

    call = client.calls[0]
    assert call["state"] == {"user_message": "hello world"}
    assert call["questions"]["intent"].criteria == {"x": None, "y": None}


def test_predict_records_error_without_raising() -> None:
    client = _FakeClient(error=TypeSafeError("simulated failure"))
    evaluator = JevEvaluator(client, experiment="unit-test")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.prediction is None
    assert result.correct is None
    assert result.confidence is None
    assert result.error == "simulated failure"


def test_predict_missing_request_id_does_not_raise() -> None:
    client = _FakeClient(response=_FakeResponse("track_order", {"track_order": 0.9}, request_id=None))
    evaluator = JevEvaluator(client, experiment="unit-test")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.request_id is None
