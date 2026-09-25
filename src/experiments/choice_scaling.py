"""Experiment C: choice scaling -- what happens as the "smart switch" gets larger?

Evaluates Jev's accuracy, latency, and confidence as the number of candidate
intents (K) grows across {5, 10, 25, 60}. The same 100 base examples (see
scripts/build_choice_scaling_sample.py; ground_truth restricted to MASSIVE's
5 most frequent en-US intents) are reused unmodified at every K -- only the
candidate list changes, each a frequency-ranked prefix of the 60 MASSIVE
intents (K=5 subset of K=10 subset of K=25 subset of K=60). Holding the
examples fixed isolates candidate-set size as the only variable.

Usage:
    uv run python src/experiments/choice_scaling.py [--concurrency N] [--output PATH]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from clients.factory import build_client  # noqa: E402
from dataset_loaders.massive import (  # noqa: E402
    CHOICE_SCALING_K_VALUES,
    DEFAULT_SEED,
    SAMPLES_DIR,
    load_sample,
    to_example,
)
from evaluation.metrics import compute_metrics  # noqa: E402
from evaluation.recorder import DEFAULT_RESULTS_PATH, append_results, load_results  # noqa: E402
from evaluation.runner import SystemOneEvaluator, run_evaluation_async  # noqa: E402
from models.experiment import ExperimentConfig  # noqa: E402

EXPERIMENT_NAME = "choice_scaling"
SAMPLE_NAME = "choice_scaling"
INTENT_ORDER_PATH = SAMPLES_DIR / "massive_choice_scaling_intent_order.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment C: Jev accuracy vs. number of candidate intents (K).")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of Jev requests in flight at once, per K (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Parquet file to append raw results to (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--providers",
        type=lambda s: [p.strip() for p in s.split(",") if p.strip()],
        default=["jev"],
        help="Comma-separated providers to run and compare, e.g. 'jev,openai:gpt-4o-mini' (default: jev).",
    )
    return parser.parse_args(argv)


def _load_subsets() -> dict[int, list[str]]:
    data = json.loads(INTENT_ORDER_PATH.read_text())
    return {int(k): v for k, v in data["subsets"].items()}


async def run(concurrency: int, output: Path, providers: list[str] | None = None) -> list[dict]:
    providers = providers or ["jev"]
    subsets = _load_subsets()
    rows = load_sample(SAMPLE_NAME)

    tidy_rows = []
    for k in CHOICE_SCALING_K_VALUES:
        candidates = subsets[k]
        missing_ground_truth = [row.id for row in rows if row.ground_truth not in candidates]
        if missing_ground_truth:
            raise ValueError(f"K={k}: {len(missing_ground_truth)} example(s) have ground_truth outside the K-choice set, e.g. {missing_ground_truth[:5]}.")
        examples = [to_example(row, candidates=candidates) for row in rows]

        for spec in providers:
            client, label = build_client(spec)
            config = ExperimentConfig(
                name=EXPERIMENT_NAME,
                dataset=f"massive:choice_scaling:k{k}",
                providers=(label,),
                seed=DEFAULT_SEED,
            )

            async with client:
                evaluator = SystemOneEvaluator(client, experiment=f"{EXPERIMENT_NAME}_k{k}", provider=label)
                results = await run_evaluation_async(evaluator, examples, concurrency=concurrency)

            append_results(results, config, path=output)

            run_df = load_results(output).query("run_id == @config.run_id")
            metrics = compute_metrics(run_df)
            confidence = run_df["confidence"].dropna()

            row = {
                "provider": label,
                "num_choices": k,
                "accuracy": metrics["accuracy"],
                "p50_latency_ms": metrics["p50_latency_ms"],
                "p95_latency_ms": metrics["p95_latency_ms"],
                "mean_confidence": float(confidence.mean()) if not confidence.empty else None,
                "n": metrics["n"],
            }
            tidy_rows.append(row)
            print(f"{label:>12} K={k:>2}: accuracy={row['accuracy']} p50={row['p50_latency_ms']:.0f}ms p95={row['p95_latency_ms']:.0f}ms conf={row['mean_confidence']} n={row['n']}")

    table_path = output.parent / f"{output.stem}_{EXPERIMENT_NAME}_table.csv"
    pd.DataFrame(tidy_rows).to_csv(table_path, index=False)
    print(f"\ntidy table: {table_path}")
    return tidy_rows


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.concurrency, args.output, args.providers))


if __name__ == "__main__":
    main()
