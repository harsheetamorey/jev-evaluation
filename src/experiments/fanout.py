"""Experiment D: parallel decision fan-out -- one state, many simultaneous questions.

Measures how Jev's `system_one` scales as the NUMBER OF QUESTIONS asked about a
single customer message grows (1, 5, 10, 25, 50), using one API call per
(message, question-count) pair rather than one call per question -- the SDK
takes a dict of Choice/Noul/Score questions in a single call, all answered
together.

This is about fan-out latency, not correctness: most of QUESTION_SPECS (e.g.
"urgent") has no ground truth, so no accuracy is computed or claimed anywhere
in this experiment -- only total latency, latency per question, and input
size (message length and reported input/output tokens).

Usage:
    uv run python src/experiments/fanout.py [--concurrency N] [--output PATH]
"""

import argparse
import asyncio
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from typesafe_sdk import Choice, Noul, Question, Score, TypeSafeError  # noqa: E402

from clients.jev_client import AsyncJevClient  # noqa: E402
from dataset_loaders.bitext import DEFAULT_SEED, load_sample  # noqa: E402
from evaluation.recorder import DEFAULT_RESULTS_PATH  # noqa: E402

EXPERIMENT_NAME = "fanout"
QUESTION_COUNTS = [1, 5, 10, 25, 50]
N_MESSAGES = 5

# Ordered so any prefix is a valid question set: QUESTION_SPECS[:1], [:5], [:10],
# [:25], [:50]. Mostly Noul (yes/no) signals a customer-support triage system
# might fan out to, plus a couple of Choice and Score questions so every
# question type the SDK supports is exercised. None of these have ground
# truth -- this experiment never scores them, only times them.
QUESTION_SPECS: list[tuple[str, Question]] = [
    ("intent", Choice(
        instructions="What is this message primarily about?",
        criteria={
            "billing": None, "technical_support": None, "account_management": None,
            "shipping": None, "general_inquiry": None, "cancellation": None,
            "complaint": None, "other": None,
        },
    )),
    ("asks_for_refund", Noul(instructions="Does the message ask for a refund?")),
    ("mentions_payment", Noul(instructions="Does the message mention a payment or charge?")),
    ("mentions_delivery", Noul(instructions="Does the message mention delivery or shipping?")),
    ("needs_account_help", Noul(instructions="Does the message need help with an account?")),
    ("explicitly_requests_human", Noul(instructions="Does the message explicitly ask for a human agent?")),
    ("negative_tone", Noul(instructions="Does the message have a negative tone?")),
    ("urgent", Noul(instructions="Does the message convey urgency?")),
    ("mentions_order_number", Noul(instructions="Does the message mention an order number?")),
    ("mentions_price", Noul(instructions="Does the message mention a price or amount?")),
    ("asks_for_cancellation", Noul(instructions="Does the message ask to cancel something?")),
    ("mentions_subscription", Noul(instructions="Does the message mention a subscription?")),
    ("mentions_login_issue", Noul(instructions="Does the message mention trouble logging in?")),
    ("asks_how_to", Noul(instructions="Does the message ask how to do something?")),
    ("mentions_bug_or_error", Noul(instructions="Does the message mention a bug or error?")),
    ("mentions_shipping_delay", Noul(instructions="Does the message mention a shipping delay?")),
    ("asks_for_discount", Noul(instructions="Does the message ask for a discount?")),
    ("mentions_competitor", Noul(instructions="Does the message mention a competitor?")),
    ("threatens_to_leave", Noul(instructions="Does the message threaten to stop being a customer?")),
    ("expresses_gratitude", Noul(instructions="Does the message express thanks?")),
    ("asks_about_return_policy", Noul(instructions="Does the message ask about the return policy?")),
    ("mentions_wrong_item", Noul(instructions="Does the message mention receiving the wrong item?")),
    ("mentions_damaged_item", Noul(instructions="Does the message mention a damaged item?")),
    ("asks_for_invoice", Noul(instructions="Does the message ask for an invoice or receipt?")),
    ("mentions_duplicate_charge", Noul(instructions="Does the message mention being charged twice?")),
    ("asks_to_update_address", Noul(instructions="Does the message ask to update a shipping address?")),
    ("asks_to_update_payment_method", Noul(instructions="Does the message ask to update a payment method?")),
    ("mentions_app_crash", Noul(instructions="Does the message mention the app crashing?")),
    ("mentions_website_issue", Noul(instructions="Does the message mention a website problem?")),
    ("asks_for_status_update", Noul(instructions="Does the message ask for a status update?")),
    ("mentions_warranty", Noul(instructions="Does the message mention a warranty?")),
    ("asks_for_manager", Noul(instructions="Does the message ask to speak with a manager?")),
    ("uses_profanity", Noul(instructions="Does the message use profanity?")),
    ("mentions_password_reset", Noul(instructions="Does the message mention resetting a password?")),
    ("mentions_two_factor_auth", Noul(instructions="Does the message mention two-factor authentication?")),
    ("asks_for_refund_status", Noul(instructions="Does the message ask about the status of a refund?")),
    ("mentions_late_delivery", Noul(instructions="Does the message mention a late delivery?")),
    ("mentions_missing_item", Noul(instructions="Does the message mention a missing item?")),
    ("asks_to_close_account", Noul(instructions="Does the message ask to close an account?")),
    ("mentions_fraud_concern", Noul(instructions="Does the message mention a fraud concern?")),
    ("asks_for_tracking_number", Noul(instructions="Does the message ask for a tracking number?")),
    ("mentions_gift_card", Noul(instructions="Does the message mention a gift card?")),
    ("asks_about_promotion", Noul(instructions="Does the message ask about a promotion or sale?")),
    ("mentions_email_not_received", Noul(instructions="Does the message mention not receiving an email?")),
    ("asks_for_callback", Noul(instructions="Does the message ask for a callback?")),
    ("tone_category", Choice(
        instructions="Which best describes the message's tone?",
        criteria={"calm": None, "frustrated": None, "angry": None, "neutral": None},
    )),
    ("urgency_level", Score(
        instructions="How urgent is this message?",
        criteria=["not urgent", "somewhat urgent", "urgent", "extremely urgent"],
    )),
    ("sentiment_level", Score(
        instructions="What is the sentiment of this message?",
        criteria=["very negative", "negative", "neutral", "positive", "very positive"],
    )),
    ("politeness_level", Score(
        instructions="How polite is this message?",
        criteria=["rude", "neutral", "polite"],
    )),
    ("complexity_level", Score(
        instructions="How complex is the request in this message?",
        criteria=["simple", "moderate", "complex"],
    )),
]

assert len(QUESTION_SPECS) == 50, f"expected 50 question specs, got {len(QUESTION_SPECS)}"


@dataclass
class FanoutResult:
    """One (message, question-count) call's fan-out performance -- no correctness fields."""

    message_id: str
    num_questions: int
    total_latency_ms: float
    latency_per_question_ms: float
    input_chars: int
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


def build_questions(n: int) -> dict[str, Question]:
    """The first `n` questions from QUESTION_SPECS, so counts nest (1 subset of 5 subset of ...)."""
    if not 1 <= n <= len(QUESTION_SPECS):
        raise ValueError(f"n must be between 1 and {len(QUESTION_SPECS)}, got {n}.")
    return dict(QUESTION_SPECS[:n])


async def run_one(client: AsyncJevClient, message_id: str, text: str, n: int) -> FanoutResult:
    questions = build_questions(n)
    start = time.perf_counter()
    try:
        response = await client.system_one(state={"user_message": text}, questions=questions)
    except TypeSafeError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return FanoutResult(
            message_id=message_id,
            num_questions=n,
            total_latency_ms=latency_ms,
            latency_per_question_ms=latency_ms / n,
            input_chars=len(text),
            input_tokens=None,
            output_tokens=None,
            error=str(exc),
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return FanoutResult(
        message_id=message_id,
        num_questions=n,
        total_latency_ms=latency_ms,
        latency_per_question_ms=latency_ms / n,
        input_chars=len(text),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        error=None,
    )


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Mean total latency / latency-per-question / tokens per question count, across messages.

    Takes a DataFrame shaped like FanoutResult rows (e.g. loaded straight from
    a saved fanout_results.parquet) -- see `summarize()` for the list-of-results
    entry point used when results are still in memory.
    """
    return (
        df.groupby("num_questions")
        .agg(
            total_latency_ms=("total_latency_ms", "mean"),
            latency_per_question_ms=("latency_per_question_ms", "mean"),
            input_tokens=("input_tokens", "mean"),
            output_tokens=("output_tokens", "mean"),
            error_rate=("error", lambda s: s.notna().mean()),
            n_messages=("message_id", "count"),
        )
        .reset_index()
        .sort_values("num_questions")
    )


def summarize(results: list[FanoutResult]) -> pd.DataFrame:
    """Mean total latency / latency-per-question / tokens per question count, across messages."""
    return aggregate_results(pd.DataFrame([asdict(r) for r in results]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment D: parallel decision fan-out latency.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of Jev requests in flight at once (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH.parent / "fanout_results.parquet",
        help="Parquet file for raw per-call fan-out results (default: data/results/fanout_results.parquet).",
    )
    return parser.parse_args(argv)


async def run(concurrency: int, output: Path) -> pd.DataFrame:
    messages = random.Random(DEFAULT_SEED).sample(load_sample("dev"), N_MESSAGES)

    semaphore = asyncio.Semaphore(concurrency)

    async def bound(client: AsyncJevClient, message_id: str, text: str, n: int) -> FanoutResult:
        async with semaphore:
            return await run_one(client, message_id, text, n)

    async with AsyncJevClient() as client:
        tasks = [bound(client, row.id, row.text, n) for row in messages for n in QUESTION_COUNTS]
        results = await asyncio.gather(*tasks)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_parquet(output, index=False)

    summary = summarize(results)
    summary_path = output.parent / f"{output.stem}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"raw results: {output} ({len(results)} calls)")
    print(f"summary: {summary_path}")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.concurrency, args.output))


if __name__ == "__main__":
    main()
