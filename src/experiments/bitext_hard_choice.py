"""Experiment A: Jev hard-choice classification on Bitext MCQ.

Usage:
    uv run python src/experiments/bitext_hard_choice.py [--sample-size {50,250,1000}] [--concurrency N] [--output PATH]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.jev_client import AsyncJevClient, JevEvaluator  # noqa: E402
from dataset_loaders.bitext import SAMPLE_SIZES, load_sample, to_example  # noqa: E402
from evaluation.metrics import compute_metrics  # noqa: E402
from evaluation.recorder import DEFAULT_RESULTS_PATH, append_results, load_results  # noqa: E402
from evaluation.runner import run_evaluation_async  # noqa: E402
from models.experiment import ExperimentConfig  # noqa: E402

EXPERIMENT_NAME = "bitext_hard_choice"
SIZE_TO_SAMPLE_NAME = {size: name for name, size in SAMPLE_SIZES.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment A: Bitext hard-choice classification via Jev.")
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=sorted(SIZE_TO_SAMPLE_NAME),
        default=50,
        help="Which pre-built Bitext sample to run against (default: 50, the dev sample).",
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

    config = ExperimentConfig(name=EXPERIMENT_NAME, dataset=f"bitext:{sample_name}", providers=("jev",))

    async with AsyncJevClient() as client:
        evaluator = JevEvaluator(client, experiment=EXPERIMENT_NAME)
        results = await run_evaluation_async(evaluator, examples, concurrency=concurrency)

    append_results(results, config, path=output)

    run_df = load_results(output).query("run_id == @config.run_id")
    metrics = compute_metrics(run_df)

    metrics_path = output.parent / f"{output.stem}_{EXPERIMENT_NAME}_metrics.json"
    metrics_path.write_text(json.dumps({"experiment": EXPERIMENT_NAME, "run_id": config.run_id, **metrics}, indent=2))

    print(f"run_id: {config.run_id}")
    print(f"results: {output}")
    print(f"metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.sample_size, args.concurrency, args.output))


if __name__ == "__main__":
    main()
