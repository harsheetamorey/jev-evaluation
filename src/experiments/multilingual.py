"""Experiment B: multilingual Jev classification on aligned MASSIVE utterances.

Each request classifies an utterance among all 60 MASSIVE intents. Because
aligned examples share their `example_id` across the 8 locales (see
dataset_loaders.massive.to_example), results can be grouped both by locale
and by aligned ID.

Usage:
    uv run python src/experiments/multilingual.py [--sample-size {100,250}] [--concurrency N] [--output PATH]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.jev_client import AsyncJevClient, JevEvaluator  # noqa: E402
from dataset_loaders.massive import ALIGNED_SAMPLE_SIZES, load_sample, to_example  # noqa: E402
from evaluation.metrics import compute_cross_language_consistency, compute_metrics, compute_metrics_by_group  # noqa: E402
from evaluation.recorder import DEFAULT_RESULTS_PATH, append_results, load_results  # noqa: E402
from evaluation.runner import run_evaluation_async  # noqa: E402
from models.experiment import ExperimentConfig  # noqa: E402

EXPERIMENT_NAME = "multilingual"
SIZE_TO_SAMPLE_NAME = {size: name for name, size in ALIGNED_SAMPLE_SIZES.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment B: multilingual Jev classification on aligned MASSIVE utterances.")
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=sorted(SIZE_TO_SAMPLE_NAME),
        default=100,
        help="Which pre-built aligned MASSIVE sample to run against (default: 100 shared IDs x 8 locales).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of Jev requests in flight at once (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Parquet file to append raw results to (default: {DEFAULT_RESULTS_PATH}).",
    )
    return parser.parse_args(argv)


async def run(sample_size: int, concurrency: int, output: Path) -> dict:
    sample_name = SIZE_TO_SAMPLE_NAME[sample_size]
    examples = [to_example(row) for row in load_sample(sample_name)]

    config = ExperimentConfig(name=EXPERIMENT_NAME, dataset=f"massive:{sample_name}", providers=("jev",))

    async with AsyncJevClient() as client:
        evaluator = JevEvaluator(client, experiment=EXPERIMENT_NAME)
        results = await run_evaluation_async(evaluator, examples, concurrency=concurrency)

    append_results(results, config, path=output)

    run_df = load_results(output).query("run_id == @config.run_id")
    summary = {
        "overall": compute_metrics(run_df),
        "by_locale": compute_metrics_by_group(run_df, "locale"),
        "cross_language_consistency": compute_cross_language_consistency(run_df),
    }

    summary_path = output.parent / f"{output.stem}_{EXPERIMENT_NAME}_metrics.json"
    summary_path.write_text(json.dumps({"experiment": EXPERIMENT_NAME, "run_id": config.run_id, **summary}, indent=2))

    print(f"run_id: {config.run_id}")
    print(f"results: {output}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["overall"], indent=2))
    print("\nby locale (accuracy / p50 / p95 / error_rate):")
    for locale, m in summary["by_locale"].items():
        print(f"  {locale:8s} {m['accuracy']}  {m['p50_latency_ms']}  {m['p95_latency_ms']}  {m['error_rate']}")
    print("\ncross-language consistency histogram (correct_locales/total_locales -> count of IDs):")
    print(json.dumps(summary["cross_language_consistency"]["histogram"], indent=2))

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.sample_size, args.concurrency, args.output))


if __name__ == "__main__":
    main()
