"""Experiment E: confidence behavior -- exploratory analysis only (Part I scope).

Compares Jev's confidence distribution for correct vs. incorrect predictions
from EXISTING recorded results (no new Jev calls here), and surfaces example
rows in four quadrants: high/low confidence crossed with correct/wrong. This
is descriptive, not calibration -- it doesn't fit or validate a probability
model, just looks at what the recorded confidence values look like next to
correctness. Full calibration is out of scope for Part I.

Usage:
    uv run python src/experiments/confidence.py \\
        [--input PATH] [--experiment NAME] [--provider NAME] \\
        [--high-threshold 0.85] [--low-threshold 0.5] [--output-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from evaluation.recorder import DEFAULT_RESULTS_PATH  # noqa: E402

EXPERIMENT_NAME = "confidence"
HIGH_CONFIDENCE_THRESHOLD = 0.85  # matches the routing-demo threshold in experiments/routing_demo.py
LOW_CONFIDENCE_THRESHOLD = 0.5
EXAMPLES_PER_QUADRANT = 3
DEFAULT_SEED = 42

EXAMPLE_COLUMNS = ["example_id", "dataset", "locale", "provider", "text", "prediction", "ground_truth", "confidence"]


def load_scored_results(path: Path, experiment: str | None = None, provider: str | None = None) -> pd.DataFrame:
    """Load recorded results, keeping only scored rows (correct and confidence both present)."""
    df = pd.read_parquet(path)
    df = df[df["correct"].notna() & df["confidence"].notna()]
    if experiment is not None:
        df = df[df["experiment"] == experiment]
    if provider is not None:
        df = df[df["provider"] == provider]
    return df


def confidence_distribution_by_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Confidence summary stats (n, mean, median, std, min, max) for correct vs. incorrect rows."""
    if df.empty:
        return pd.DataFrame(columns=["correct", "n", "mean", "median", "std", "min", "max"])
    return (
        df.groupby("correct")["confidence"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )


def confidently_wrong(df: pd.DataFrame, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Incorrect predictions made with confidence >= threshold -- the interesting failure mode."""
    return df[df["correct"].eq(False) & (df["confidence"] >= threshold)]


def low_confidence_examples(df: pd.DataFrame, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """All rows (correct or not) with confidence below threshold."""
    return df[df["confidence"] < threshold]


def quadrant_examples(
    df: pd.DataFrame,
    high: float = HIGH_CONFIDENCE_THRESHOLD,
    low: float = LOW_CONFIDENCE_THRESHOLD,
    n: int = EXAMPLES_PER_QUADRANT,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Up to `n` deterministically-sampled example rows in each of 4 quadrants."""
    selectors = {
        "high_confidence_correct": (df["confidence"] >= high) & df["correct"].eq(True),
        "high_confidence_wrong": (df["confidence"] >= high) & df["correct"].eq(False),
        "low_confidence_correct": (df["confidence"] < low) & df["correct"].eq(True),
        "low_confidence_wrong": (df["confidence"] < low) & df["correct"].eq(False),
    }
    columns = [c for c in EXAMPLE_COLUMNS if c in df.columns]
    result = {}
    for name, mask in selectors.items():
        subset = df.loc[mask, columns]
        if len(subset) > n:
            subset = subset.sample(n=n, random_state=seed)
        result[name] = subset
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment E: confidence-vs-correctness exploratory analysis (Part I -- not calibration)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS_PATH, help=f"Results Parquet to analyze (default: {DEFAULT_RESULTS_PATH}).")
    parser.add_argument("--experiment", type=str, default=None, help="Restrict to one experiment name (default: all).")
    parser.add_argument("--provider", type=str, default=None, help="Restrict to one provider (default: all).")
    parser.add_argument("--high-threshold", type=float, default=HIGH_CONFIDENCE_THRESHOLD, help=f"'High confidence' cutoff (default: {HIGH_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--low-threshold", type=float, default=LOW_CONFIDENCE_THRESHOLD, help=f"'Low confidence' cutoff (default: {LOW_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_PATH.parent, help="Directory to write confidence_summary.json into.")
    return parser.parse_args(argv)


def run(
    input_path: Path,
    experiment: str | None,
    provider: str | None,
    high: float,
    low: float,
    output_dir: Path,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist yet -- run an experiment (e.g. bitext_hard_choice.py) first to produce labeled results.")

    df = load_scored_results(input_path, experiment=experiment, provider=provider)
    if df.empty:
        raise ValueError("No scored rows (both 'correct' and 'confidence' populated) found for the given filters.")

    distribution = confidence_distribution_by_correctness(df)
    wrong_confident = confidently_wrong(df, threshold=high)
    low_conf = low_confidence_examples(df, threshold=low)
    quadrants = quadrant_examples(df, high=high, low=low)

    summary = {
        "n_total": len(df),
        "high_confidence_threshold": high,
        "low_confidence_threshold": low,
        "distribution_by_correctness": distribution.to_dict(orient="records"),
        "n_confidently_wrong": len(wrong_confident),
        "confidently_wrong_rate": len(wrong_confident) / len(df),
        "n_low_confidence": len(low_conf),
        "quadrant_examples": {name: rows.to_dict(orient="records") for name, rows in quadrants.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "confidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps({k: v for k, v in summary.items() if k != "quadrant_examples"}, indent=2, default=str))
    print("\nquadrant examples:")
    for name, rows in quadrants.items():
        print(f"\n  {name.upper()} ({len(rows)} shown):")
        for _, r in rows.iterrows():
            text = (r.get("text") or "")[:80]
            print(f"    [conf={r['confidence']:.2f}] pred={r['prediction']!r} truth={r['ground_truth']!r} text={text!r}")
    print(f"\nsaved: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run(args.input, args.experiment, args.provider, args.high_threshold, args.low_threshold, args.output_dir)


if __name__ == "__main__":
    main()
