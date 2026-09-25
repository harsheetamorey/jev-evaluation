"""Experiment B: multilingual classification on aligned MASSIVE utterances, one or more providers.

Each request classifies an utterance among all 60 MASSIVE intents. Because
aligned examples share their `example_id` across locales (see
dataset_loaders.massive.to_example), results can be grouped by locale and by
aligned ID. Runs each requested provider sequentially over the same
examples, so results are directly comparable -- an LLM provider is always
identified by its exact model name, never a generic "llm" label.

Usage:
    uv run python src/experiments/multilingual.py \\
        [--sample-size {100,250}] [--concurrency N] [--output PATH] \\
        [--locales en-US,hi-IN] [--providers jev,openai:gpt-4o-mini]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.factory import build_client  # noqa: E402
from dataset_loaders.massive import ALIGNED_SAMPLE_SIZES, LOCALES, load_sample, to_example  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_cross_language_consistency,
    compute_metrics,
    compute_metrics_by_group,
    format_comparison_table,
)
from evaluation.recorder import DEFAULT_RESULTS_PATH, append_results, load_results  # noqa: E402
from evaluation.runner import SystemOneEvaluator, run_evaluation_async  # noqa: E402
from models.experiment import ExperimentConfig  # noqa: E402
from models.prediction import Example, PredictionResult  # noqa: E402

EXPERIMENT_NAME = "multilingual"
SIZE_TO_SAMPLE_NAME = {size: name for name, size in ALIGNED_SAMPLE_SIZES.items()}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment B: multilingual classification on aligned MASSIVE utterances.")
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=sorted(SIZE_TO_SAMPLE_NAME),
        default=100,
        help="Which pre-built aligned MASSIVE sample to run against (default: 100 shared IDs).",
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
        "--locales",
        type=_split_csv,
        default=list(LOCALES),
        help=f"Comma-separated subset of locales to run (default: all {len(LOCALES)}: {','.join(LOCALES)}).",
    )
    parser.add_argument(
        "--providers",
        type=_split_csv,
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


async def run(sample_size: int, concurrency: int, output: Path, locales: list[str], providers: list[str]) -> dict:
    unknown_locales = set(locales) - set(LOCALES)
    if unknown_locales:
        raise ValueError(f"Unknown locale(s) {sorted(unknown_locales)}; expected a subset of {LOCALES}.")

    sample_name = SIZE_TO_SAMPLE_NAME[sample_size]
    rows = [row for row in load_sample(sample_name) if row.locale in locales]
    examples = [to_example(row) for row in rows]

    provider_labels: list[str] = []
    all_results: list[PredictionResult] = []
    for spec in providers:
        label, results = await run_provider(spec, examples, concurrency)
        provider_labels.append(label)
        all_results.extend(results)

    config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        dataset=f"massive:{sample_name}:{'+'.join(locales)}",
        providers=tuple(provider_labels),
    )
    append_results(all_results, config, path=output)

    run_df = load_results(output).query("run_id == @config.run_id")
    by_provider = compute_metrics_by_group(run_df, "provider")
    summary = {
        "overall": compute_metrics(run_df),
        "by_provider": by_provider,
        # Per provider, then per locale -- pooling providers into one per-locale
        # number would average Jev and an LLM together and report it as "the"
        # accuracy for that language.
        "by_locale": {
            provider: compute_metrics_by_group(provider_df, "locale")
            for provider, provider_df in run_df.groupby("provider")
        },
        "cross_language_consistency": {
            provider: compute_cross_language_consistency(provider_df)
            for provider, provider_df in run_df.groupby("provider")
        },
    }

    summary_path = output.parent / f"{output.stem}_{EXPERIMENT_NAME}_metrics.json"
    summary_path.write_text(json.dumps({"experiment": EXPERIMENT_NAME, "run_id": config.run_id, **summary}, indent=2))

    print(f"run_id: {config.run_id}")
    print(f"results: {output}")
    print(f"summary: {summary_path}")
    print()
    print(format_comparison_table(by_provider))
    print("\nby locale (accuracy / p50 ms / error rate):")
    for provider, locales in summary["by_locale"].items():
        print(f"  {provider}:")
        for locale, m in sorted(locales.items()):
            print(f"    {locale:8s} {m['accuracy']:.3f}  {m['p50_latency_ms']:>7.0f}  {m['error_rate']:.3f}")
    print("\ncross-language consistency histogram per provider (correct_locales/total_locales -> count of IDs):")
    for provider, consistency in summary["cross_language_consistency"].items():
        print(f"  {provider}: {json.dumps(consistency['histogram'])}")

    return summary


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.sample_size, args.concurrency, args.output, args.locales, args.providers))


if __name__ == "__main__":
    main()
