"""Unit tests for the contract evolution demo -- no network/keys needed."""

import asyncio

import pytest

from experiments.contract_evolution import (
    AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS,
    EXPANDED_CONTRACT,
    SMALL_CONTRACT,
    classify,
)


class _FakeAnswer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice = choice
        self.probabilities = probabilities


class _FakeResponse:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choices = {"intent": _FakeAnswer(choice, probabilities)}


class _FakeClient:
    def __init__(self, choice: str) -> None:
        self._choice = choice
        self.last_questions = None

    async def system_one(self, state, questions):
        self.last_questions = questions
        return _FakeResponse(self._choice, {self._choice: 0.9})


def test_classify_returns_the_answer_when_choice_is_valid() -> None:
    client = _FakeClient(choice="billing")
    answer = asyncio.run(classify(client, "I was charged twice", SMALL_CONTRACT))
    assert answer.choice == "billing"


def test_classify_raises_on_contract_violation() -> None:
    client = _FakeClient(choice="not_a_real_choice")
    with pytest.raises(AssertionError, match="contract violation"):
        asyncio.run(classify(client, "some message", SMALL_CONTRACT))


def test_classify_sends_criteria_matching_the_given_choices() -> None:
    client = _FakeClient(choice="other")
    asyncio.run(classify(client, "some message", EXPANDED_CONTRACT))
    assert set(client.last_questions["intent"].criteria) == set(EXPANDED_CONTRACT)


def test_expanded_contract_is_a_superset_of_small_contract() -> None:
    assert set(SMALL_CONTRACT) < set(EXPANDED_CONTRACT)


def test_shipping_only_exists_in_the_expanded_contract() -> None:
    assert "shipping" not in SMALL_CONTRACT
    assert "shipping" in EXPANDED_CONTRACT


def test_ambiguous_message_plausible_labels_are_both_in_expanded_contract() -> None:
    assert AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS <= set(EXPANDED_CONTRACT)
    assert len(AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS) >= 2
