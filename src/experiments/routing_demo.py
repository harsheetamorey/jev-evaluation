"""Experiment F: confidence-based routing demo.

Demonstrates a tiny application decision built on top of Jev's confidence:

    Jev -> confidence >= threshold?
             yes -> execute_route(intent)   (normal code, fast path)
             no  -> fallback_to_llm()       (LLM, slower safety net)

The educational point: confidence can affect what software does next, not
just what gets logged. Part I scope only -- the threshold (0.85 by default)
is NOT tuned or validated here; that calibration work belongs in Part II.

Usage:
    uv run python src/experiments/routing_demo.py "I want to cancel my order" \\
        [--threshold 0.85] [--candidates track_order,cancel_order,payment_issue,contact_support,other] \\
        [--fallback-llm openai:gpt-4o-mini]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.factory import build_client  # noqa: E402
from clients.jev_client import AsyncJevClient  # noqa: E402
from evaluation.runner import SystemOneEvaluator  # noqa: E402
from models.prediction import Example  # noqa: E402

EXPERIMENT_NAME = "routing_demo"
DEFAULT_THRESHOLD = 0.85
DEFAULT_CANDIDATES = ["track_order", "cancel_order", "payment_issue", "contact_support", "other"]


def decide_route(confidence: float | None, threshold: float) -> str:
    """Pure routing decision: does Jev's confidence clear the threshold?

    A missing confidence (e.g. Jev itself errored) is treated as not
    clearing it -- fail toward the LLM fallback, never toward blindly
    executing a route with no confidence behind it.
    """
    if confidence is None:
        return "fallback_to_llm"
    return "execute_route" if confidence >= threshold else "fallback_to_llm"


def execute_route(intent: str) -> str:
    """Stand-in for 'normal code': the fast path taken when confidence clears the bar.

    A real system would dispatch to a real handler per intent; this demo
    only describes the action, keeping the routing behavior itself the
    focus rather than any particular business logic.
    """
    return f"ROUTE -> {intent}"


async def run(message: str, threshold: float, candidates: list[str], fallback_spec: str | None) -> dict:
    example = Example(example_id="demo", dataset="demo", state=message, candidates=candidates)

    async with AsyncJevClient() as jev_client:
        jev_evaluator = SystemOneEvaluator(jev_client, experiment=EXPERIMENT_NAME, provider="jev")
        jev_result = await jev_evaluator.predict(example)

    route = decide_route(jev_result.confidence, threshold)
    outcome = {
        "message": message,
        "threshold": threshold,
        "jev_prediction": jev_result.prediction,
        "jev_confidence": jev_result.confidence,
        "jev_error": jev_result.error,
        "route": route,
    }

    if route == "execute_route":
        outcome["software_action"] = execute_route(jev_result.prediction)
    else:
        if fallback_spec is None:
            raise ValueError(
                "Jev's confidence didn't clear the threshold (or Jev errored) and no "
                "--fallback-llm was given; pass e.g. --fallback-llm openai:gpt-4o-mini."
            )
        llm_client, llm_label = build_client(fallback_spec)
        async with llm_client:
            llm_evaluator = SystemOneEvaluator(llm_client, experiment=EXPERIMENT_NAME, provider=llm_label)
            llm_result = await llm_evaluator.predict(example)
        outcome["llm_provider"] = llm_label
        outcome["llm_prediction"] = llm_result.prediction
        outcome["software_action"] = execute_route(llm_result.prediction)

    print(json.dumps(outcome, indent=2))
    return outcome


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment F: confidence-based routing demo (Jev fast path, LLM fallback).")
    parser.add_argument("message", type=str, help="Customer message to route.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Confidence routing threshold (default: {DEFAULT_THRESHOLD}, not tuned).")
    parser.add_argument(
        "--candidates",
        type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
        default=DEFAULT_CANDIDATES,
        help=f"Comma-separated candidate intents (default: {','.join(DEFAULT_CANDIDATES)}).",
    )
    parser.add_argument(
        "--fallback-llm",
        type=str,
        default=None,
        help="Provider spec for the LLM fallback, e.g. 'openai:gpt-4o-mini' -- only needed if confidence is ever low.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.message, args.threshold, args.candidates, args.fallback_llm))


if __name__ == "__main__":
    main()
