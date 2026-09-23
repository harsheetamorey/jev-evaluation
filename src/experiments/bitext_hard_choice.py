"""Experiment A: hard-choice classification on Bitext MCQ, one or more providers.

Runs each requested provider sequentially (Jev, and/or an LLM via TypeSafe's
System One Adapter) over the same sample, so results are directly comparable.
An LLM provider is always identified by its exact model name (e.g.
"gpt-4o-mini"), never a generic "llm" label -- see clients.factory.

Usage:
    uv run python src/experiments/bitext_hard_choice.py \\
        [--sample-size {50,250,1000}] [--concurrency N] [--output PATH] \\
        [--providers jev,openai:gpt-4o-mini,gemini:gemini-2.0-flash]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.factory import build_client  # noqa: E402
from dataset_loaders.bitext import SAMPLE_SIZES, load_sample, to_example  # noqa: E402
from evaluation.metrics import compute_metrics, compute_metrics_by_group, format_comparison_table  # noqa: E402
from evaluation.recorder import DEFAULT_RESULTS_PATH, append_results, load_results  # noqa: E402
from evaluation.runner import SystemOneEvaluator, run_evaluation_async  # noqa: E402
from models.experiment import ExperimentConfig  # noqa: E402
from models.prediction import Example, PredictionResult  # noqa: E402

EXPERIMENT_NAME = "bitext_hard_choice"
SIZE_TO_SAMPLE_NAME = {size: name for name, size in SAMPLE_SIZES.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment A: Bitext hard-choice classification.")
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
        help="Maximum number of requests in flight at once, per provider (default: 5).",
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
        help=(
            "Comma-separated providers to run and compare, e.g. "
            "'jev,openai:gpt-4o-mini,gemini:gemini-2.0-flash' (default: jev)."
        ),
    )
    return parser.parse_args(argv)


async def run_provider(spec: str, examples: list[Example], concurrency: int) -> tuple[str, list[PredictionResult]]:
    client, label = build_client(spec)
    async with client:
        evaluator = SystemOneEvaluator(client, experiment=EXPERIMENT_NAME, provider=label)
        results = await run_evaluation_async(evaluator, examples, concurrency=concurrency)
    return label, results


async def run(sample_size: int, concurrency: int, output: Path, providers: list[str]) -> dict:
    sample_name = SIZE_TO_SAMPLE_NAME[sample_size]
    examples = [to_example(row) for row in load_sample(sample_name)]

    provider_labels: list[str] = []
    all_results: list[PredictionResult] = []
    for spec in providers:
        label, results = await run_provider(spec, examples, concurrency)
        provider_labels.append(label)
        all_results.extend(results)

    config = ExperimentConfig(name=EXPERIMENT_NAME, dataset=f"bitext:{sample_name}", providers=tuple(provider_labels))
    append_results(all_results, config, path=output)

    run_df = load_results(output).query("run_id == @config.run_id")
    by_provider = compute_metrics_by_group(run_df, "provider")
    summary = {"overall": compute_metrics(run_df), "by_provider": by_provider}

    metrics_path = output.parent / f"{output.stem}_{EXPERIMENT_NAME}_metrics.json"
    metrics_path.write_text(json.dumps({"experiment": EXPERIMENT_NAME, "run_id": config.run_id, **summary}, indent=2))

    print(f"run_id: {config.run_id}")
    print(f"results: {output}")
    print(f"metrics: {metrics_path}")
    print()
    print(format_comparison_table(by_provider))
    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.sample_size, args.concurrency, args.output, args.providers))


if __name__ == "__main__":
    main()
