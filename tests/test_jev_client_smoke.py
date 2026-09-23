"""Smoke test: verifies `JevClient` can reach TypeSafe's API end-to-end."""

import pytest
from typesafe_sdk import Choice

from clients.jev_client import JevClient
from config import settings


def test_jev_client_smoke() -> None:
    if not settings.typesafe_api_key:
        pytest.skip("TYPESAFE_API_KEY not set; skipping live Jev smoke test")

    with JevClient() as client:
        response = client.system_one(
            state="I was charged twice for the same order. Please refund me.",
            questions={
                "topic": Choice(
                    instructions="What is this message about?",
                    criteria={"billing": None, "technical": None, "other": None},
                ),
            },
        )

    answer = response.choices["topic"]
    assert answer.choice in {"billing", "technical", "other"}
    assert 0.0 <= answer.probabilities[answer.choice] <= 1.0
