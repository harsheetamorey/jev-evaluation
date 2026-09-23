"""Unit tests for the fan-out experiment's pure logic -- no network/keys needed."""

import asyncio

import pytest
from typesafe_sdk import Choice, Noul, Score, TypeSafeError

from experiments.fanout import QUESTION_SPECS, FanoutResult, build_questions, run_one, summarize


def test_question_specs_has_exactly_50_uniquely_named_questions() -> None:
    assert len(QUESTION_SPECS) == 50
    names = [name for name, _ in QUESTION_SPECS]
    assert len(names) == len(set(names))


def test_question_specs_covers_all_three_question_types() -> None:
    types = {type(question) for _, question in QUESTION_SPECS}
    assert types == {Choice, Noul, Score}


@pytest.mark.parametrize("n", [1, 5, 10, 25, 50])
def test_build_questions_returns_exactly_n(n: int) -> None:
    assert len(build_questions(n)) == n


def test_build_questions_counts_are_nested_prefixes() -> None:
    names_5 = list(build_questions(5))
    names_10 = list(build_questions(10))
    assert names_5 == names_10[:5]


@pytest.mark.parametrize("n", [0, 51, -1])
def test_build_questions_rejects_out_of_range_n(n: int) -> None:
    with pytest.raises(ValueError):
        build_questions(n)


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 20) -> None:
        self.usage = _FakeUsage(input_tokens, output_tokens)


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


def test_run_one_computes_latency_per_question() -> None:
    client = _FakeClient(response=_FakeResponse())
    result = asyncio.run(run_one(client, "msg-1", "hello", 5))

    assert result.num_questions == 5
    assert result.latency_per_question_ms == pytest.approx(result.total_latency_ms / 5)
    assert result.input_chars == len("hello")
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.error is None


def test_run_one_sends_exactly_n_questions() -> None:
    client = _FakeClient(response=_FakeResponse())
    asyncio.run(run_one(client, "msg-1", "hello", 10))

    assert len(client.calls[0]["questions"]) == 10


def test_run_one_records_error_without_raising() -> None:
    client = _FakeClient(error=TypeSafeError("simulated failure"))
    result = asyncio.run(run_one(client, "msg-1", "hello", 5))

    assert result.error == "simulated failure"
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_summarize_aggregates_mean_and_error_rate_per_question_count() -> None:
    results = [
        FanoutResult("m1", 5, total_latency_ms=100, latency_per_question_ms=20, input_chars=10, input_tokens=50, output_tokens=5, error=None),
        FanoutResult("m2", 5, total_latency_ms=200, latency_per_question_ms=40, input_chars=10, input_tokens=60, output_tokens=6, error="boom"),
        FanoutResult("m1", 10, total_latency_ms=150, latency_per_question_ms=15, input_chars=10, input_tokens=80, output_tokens=8, error=None),
    ]

    summary = summarize(results)

    row_5 = summary[summary["num_questions"] == 5].iloc[0]
    assert row_5["total_latency_ms"] == pytest.approx(150.0)
    assert row_5["error_rate"] == pytest.approx(0.5)
    assert row_5["n_messages"] == 2

    row_10 = summary[summary["num_questions"] == 10].iloc[0]
    assert row_10["n_messages"] == 1
    assert row_10["error_rate"] == 0.0
