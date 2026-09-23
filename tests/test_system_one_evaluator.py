"""Unit tests for SystemOneEvaluator's mapping from a System One response to PredictionResult.

Uses a fake client (no network, no API key) so these run without credentials.
The same evaluator is used for both Jev and any LLM run through TypeSafe's
System One Adapter, so these tests aren't provider-specific.
"""

import asyncio

from typesafe_sdk import TypeSafeError

from evaluation.runner import SystemOneEvaluator
from models.prediction import Example


class _FakeAnswer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice = choice
        self.probabilities = probabilities


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5, input_tokens_total=None, output_tokens_total=None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        # Only set when present -- mirrors the adapter's Usage subclass vs Jev's plain Usage.
        if input_tokens_total is not None:
            self.input_tokens_total = input_tokens_total
        if output_tokens_total is not None:
            self.output_tokens_total = output_tokens_total


class _FakeResponse:
    def __init__(
        self,
        choice: str,
        probabilities: dict[str, float],
        request_id: str | None = "req-abc",
        model: str = "fake-model",
        usage: _FakeUsage | None = None,
    ) -> None:
        self.choices = {"intent": _FakeAnswer(choice, probabilities)}
        self._request_id = request_id
        self.model = model
        self.usage = usage if usage is not None else _FakeUsage()

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
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.provider == "jev"
    assert result.prediction == "track_order"
    assert result.correct is True
    assert result.confidence == 0.8
    assert result.error is None
    assert result.request_id == "req-abc"
    assert result.candidates == ["track_order", "cancel_order", "other"]
    assert result.text == "Where is my order?"


def test_predict_uses_provider_as_the_explicit_model_label() -> None:
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="gpt-4o-mini")

    result = asyncio.run(evaluator.predict(_example(candidates=["x", "y"])))

    assert result.provider == "gpt-4o-mini"


def test_predict_marks_incorrect_when_choice_mismatches_ground_truth() -> None:
    client = _FakeClient(response=_FakeResponse("other", {"other": 0.5}))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example(ground_truth="track_order")))

    assert result.prediction == "other"
    assert result.correct is False


def test_predict_sends_state_and_criteria_built_from_candidates() -> None:
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    asyncio.run(evaluator.predict(_example(state="hello world", candidates=["x", "y"])))

    call = client.calls[0]
    assert call["state"] == {"user_message": "hello world"}
    assert call["questions"]["intent"].criteria == {"x": None, "y": None}


def test_predict_records_error_without_raising() -> None:
    client = _FakeClient(error=TypeSafeError("simulated failure"))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.prediction is None
    assert result.correct is None
    assert result.confidence is None
    assert result.error == "simulated failure"
    assert result.text == "Where is my order?"


def test_predict_missing_request_id_does_not_raise() -> None:
    client = _FakeClient(response=_FakeResponse("track_order", {"track_order": 0.9}, request_id=None))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example()))

    assert result.request_id is None


def test_predict_records_tokens_from_plain_usage() -> None:
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}, usage=_FakeUsage(input_tokens=12, output_tokens=7)))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example(candidates=["x", "y"])))

    assert result.input_tokens == 12
    assert result.output_tokens == 7


def test_predict_prefers_cumulative_tokens_when_present() -> None:
    usage = _FakeUsage(input_tokens=12, output_tokens=7, input_tokens_total=30, output_tokens_total=15)
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}, usage=usage))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="gpt-4o-mini")

    result = asyncio.run(evaluator.predict(_example(candidates=["x", "y"])))

    assert result.input_tokens == 30
    assert result.output_tokens == 15


def test_predict_cost_is_none_without_a_pricing_entry() -> None:
    client = _FakeClient(response=_FakeResponse("x", {"x": 1.0}, model="some-unpriced-model"))
    evaluator = SystemOneEvaluator(client, experiment="unit-test", provider="jev")

    result = asyncio.run(evaluator.predict(_example(candidates=["x", "y"])))

    assert result.estimated_cost_usd is None
