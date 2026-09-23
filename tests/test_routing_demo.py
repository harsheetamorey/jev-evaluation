"""Unit tests for the confidence-based routing demo -- no network/keys needed."""

import asyncio

import pytest
from typesafe_sdk import TypeSafeError

from experiments.routing_demo import decide_route, execute_route, run


def test_decide_route_above_threshold_executes() -> None:
    assert decide_route(0.9, threshold=0.85) == "execute_route"


def test_decide_route_at_threshold_executes() -> None:
    assert decide_route(0.85, threshold=0.85) == "execute_route"


def test_decide_route_below_threshold_falls_back() -> None:
    assert decide_route(0.5, threshold=0.85) == "fallback_to_llm"


def test_decide_route_missing_confidence_falls_back() -> None:
    assert decide_route(None, threshold=0.85) == "fallback_to_llm"


def test_execute_route_describes_the_action() -> None:
    assert execute_route("track_order") == "ROUTE -> track_order"


class _FakeAnswer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice = choice
        self.probabilities = probabilities


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeResponse:
    def __init__(self, choice: str, probabilities: dict[str, float], model: str = "fake-model") -> None:
        self.choices = {"intent": _FakeAnswer(choice, probabilities)}
        self.usage = _FakeUsage()
        self.model = model

    @property
    def request_id(self):
        raise TypeSafeError("no request id")


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def system_one(self, state, questions):
        self.called = True
        return self._response


def test_run_high_confidence_never_touches_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    jev_client = _FakeClient(_FakeResponse("track_order", {"track_order": 0.95}))
    monkeypatch.setattr("experiments.routing_demo.AsyncJevClient", lambda: jev_client)

    llm_build_called = False

    def _fail_if_called(spec):
        nonlocal llm_build_called
        llm_build_called = True
        raise AssertionError("LLM should not be built on the high-confidence path")

    monkeypatch.setattr("experiments.routing_demo.build_client", _fail_if_called)

    outcome = asyncio.run(run("wake me up at 5am", threshold=0.85, candidates=["track_order", "other"], fallback_spec=None))

    assert outcome["route"] == "execute_route"
    assert outcome["software_action"] == "ROUTE -> track_order"
    assert llm_build_called is False


def test_run_low_confidence_falls_back_to_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    jev_client = _FakeClient(_FakeResponse("other", {"other": 0.3}))
    monkeypatch.setattr("experiments.routing_demo.AsyncJevClient", lambda: jev_client)

    llm_client = _FakeClient(_FakeResponse("track_order", {"track_order": 0.7}))
    monkeypatch.setattr("experiments.routing_demo.build_client", lambda spec: (llm_client, "gpt-4o-mini"))

    outcome = asyncio.run(
        run("wake me up at 5am", threshold=0.85, candidates=["track_order", "other"], fallback_spec="openai:gpt-4o-mini")
    )

    assert outcome["route"] == "fallback_to_llm"
    assert outcome["llm_provider"] == "gpt-4o-mini"
    assert outcome["software_action"] == "ROUTE -> track_order"
    assert llm_client.called is True


def test_run_low_confidence_without_fallback_spec_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    jev_client = _FakeClient(_FakeResponse("other", {"other": 0.3}))
    monkeypatch.setattr("experiments.routing_demo.AsyncJevClient", lambda: jev_client)

    with pytest.raises(ValueError, match="no --fallback-llm was given"):
        asyncio.run(run("wake me up at 5am", threshold=0.85, candidates=["track_order", "other"], fallback_spec=None))
