"""Experiment G: contract evolution demo.

Two educational points:

1. Format reliability: whatever choice list ("contract") you give Jev, the
   returned answer is always one of those exact choices -- application code
   can trust it without validating it, and this script asserts that, not
   just prints it.

2. Format reliability != decision correctness: obeying the contract does not
   guarantee the RIGHT label was chosen. AMBIGUOUS_MESSAGE deliberately
   touches two plausible categories under the expanded contract, so a
   wrong-but-contractually-valid pick is a realistic outcome to watch for,
   not a staged one -- the script reports honestly on whichever label Jev
   actually returns.

Part I scope: demonstrate the distinction, not fix it (routing on
confidence, calibration, etc. belong elsewhere / Part II).

Usage:
    uv run python src/experiments/contract_evolution.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typesafe_sdk import Choice, ChoiceAnswer  # noqa: E402

from clients.jev_client import AsyncJevClient  # noqa: E402

# Contract v1: the original, small software contract.
SMALL_CONTRACT = ["billing", "technical", "other"]

# Contract v2: evolved to add categories v1 couldn't represent.
EXPANDED_CONTRACT = ["billing", "technical", "account", "shipping", "other"]

MESSAGES_FOR_SMALL_CONTRACT = [
    "I was charged twice for my subscription this month",
    "The app crashes every time I try to open it",
    "Do you ship to Canada?",
]

# No good home in v1 (no "shipping" option) -- gets one once the contract expands.
CONTRACT_GROWTH_MESSAGE = "I need to update my shipping address before my order ships"

# Deliberately touches two plausible categories under the expanded contract:
# an account-access problem AND a billing document. A wrong-but-valid pick
# here is a realistic outcome, not a fabricated one.
AMBIGUOUS_MESSAGE = "I can't log into my account to check my last invoice"
AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS = {"account", "billing"}


async def classify(client: AsyncJevClient, message: str, choices: list[str]) -> ChoiceAnswer:
    """Ask Jev to pick one of `choices` for `message`; asserts the contract was obeyed."""
    response = await client.system_one(
        state={"user_message": message},
        questions={
            "intent": Choice(
                instructions="Select the category that best represents the user's request.",
                criteria={choice: None for choice in choices},
            )
        },
    )
    answer = response.choices["intent"]
    assert answer.choice in choices, f"contract violation: {answer.choice!r} not in {choices!r}"
    return answer


async def run() -> dict:
    async with AsyncJevClient() as client:
        print("=== Contract v1 ===")
        print(f"choices = {SMALL_CONTRACT}")
        v1_answers = {}
        for message in MESSAGES_FOR_SMALL_CONTRACT:
            answer = await classify(client, message, SMALL_CONTRACT)
            v1_answers[message] = answer.choice
            print(f"  {message!r}\n    -> {answer.choice!r}")

        print(f"\n  Contract-growth case: {CONTRACT_GROWTH_MESSAGE!r}")
        growth_v1 = await classify(client, CONTRACT_GROWTH_MESSAGE, SMALL_CONTRACT)
        print(f"    under v1 (3 choices) -> {growth_v1.choice!r}  (forced into a coarser bucket -- 'shipping' didn't exist yet)")

        print("\n=== Contract v2 ===")
        print(f"choices = {EXPANDED_CONTRACT}")
        growth_v2 = await classify(client, CONTRACT_GROWTH_MESSAGE, EXPANDED_CONTRACT)
        print(f"  {CONTRACT_GROWTH_MESSAGE!r}\n    under v2 (5 choices) -> {growth_v2.choice!r}")
        print("  Application code receives one of the 5 allowed choices either way -- the contract, not the")
        print("  application, is what changed. No parsing/validation code needed updating for the new category.")

        print("\n=== Format reliability vs. decision correctness ===")
        print(f"message: {AMBIGUOUS_MESSAGE!r}")
        print(f"plausible valid labels a human would pick: {sorted(AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS)}")
        ambiguous_answer = await classify(client, AMBIGUOUS_MESSAGE, EXPANDED_CONTRACT)
        obeyed_contract = ambiguous_answer.choice in EXPANDED_CONTRACT
        matched_expectation = ambiguous_answer.choice in AMBIGUOUS_MESSAGE_PLAUSIBLE_LABELS
        confidence = ambiguous_answer.probabilities.get(ambiguous_answer.choice)
        print(f"  Jev chose: {ambiguous_answer.choice!r} (confidence={confidence})")
        print(f"  format reliability: {'PASS' if obeyed_contract else 'FAIL'} -- always one of the given choices")
        if matched_expectation:
            print(
                "  Jev landed on a label a human would plausibly pick here -- but that's not guaranteed by the "
                "contract, only by the model being right on this particular message. Obeying the schema and "
                "being correct are two separate properties; this run just didn't happen to expose the gap."
            )
        else:
            print(
                "  Jev's choice is NOT among the labels a human would plausibly pick for this message -- while "
                "still being a perfectly valid, contract-obeying answer. This is the point: format reliability "
                "does not imply decision correctness."
            )

    return {
        "v1_answers": v1_answers,
        "contract_growth_v1": growth_v1.choice,
        "contract_growth_v2": growth_v2.choice,
        "ambiguous_choice": ambiguous_answer.choice,
        "ambiguous_matched_expectation": matched_expectation,
    }


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
