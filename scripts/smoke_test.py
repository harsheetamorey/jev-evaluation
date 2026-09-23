"""Standalone smoke test: one live Jev call, with latency logging.

Usage:
    uv run python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from typesafe_sdk import Choice  # noqa: E402

from clients.jev_client import JevClient  # noqa: E402

MESSAGE = "I ordered something last week and want to know where it is."

INTENTS = {
    "track_order": None,
    "cancel_order": None,
    "payment_issue": None,
    "contact_support": None,
    "other": None,
}


def main() -> None:
    with JevClient() as client:
        start = time.perf_counter()
        response = client.system_one(
            state={"message": MESSAGE},
            questions={
                "intent": Choice(
                    instructions="Choose the intent that best matches the message.",
                    criteria=INTENTS,
                )
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000

    answer = response.choices["intent"]
    print(f"selected choice: {answer.choice}")
    print(f"complete typed answer: {answer!r}")
    print(f"request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
