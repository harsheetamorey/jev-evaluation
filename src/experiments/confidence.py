"""Experiment E: confidence behavior -- exploratory analysis only (Part I scope).

Compares Jev's confidence distribution for correct vs. incorrect predictions
from EXISTING recorded results (no new Jev calls here), and surfaces example
rows in four quadrants: high/low confidence crossed with correct/wrong. This
is descriptive, not calibration -- it doesn't fit or validate a probability
model, just looks at what the recorded confidence values look like next to
correctness. Full calibration is out of scope for Part I.

Usage:
    uv run python src/experiments/confidence.py \\
        [--input PATH] [--experiment NAME] [--provider NAME] [--run-id ID] \\
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

# Coverage/accuracy sweep: at each threshold, what fraction of predictions clear
# it (coverage), and how accurate are they? Distinct from HIGH_CONFIDENCE_THRESHOLD,
# which flags a single specific "confidently wrong" cutoff.
DEFAULT_ACCURACY_THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
DEFAULT_HISTOGRAM_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

EXAMPLE_COLUMNS = ["example_id", "dataset", "locale", "provider", "text", "prediction", "ground_truth", "confidence"]


def load_scored_results(
    path: Path,
    experiment: str | None = None,
    provider: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Load recorded results, keeping only scored rows (correct and confidence both present).

    `run_id` is the precise way to isolate one specific run (e.g. one sample
    size among several sharing the same `experiment` name) -- `dataset` on
    each row is the dataset *family* (e.g. "bitext"), not the specific sample,
    so it can't disambiguate a 50- vs. 250- vs. 1000-example run of the same
    experiment the way `run_id` can.
    """
    df = pd.read_parquet(path)
    df = df[df["correct"].notna() & df["confidence"].notna()]
    if experiment is not None:
        df = df[df["experiment"] == experiment]
    if provider is not None:
        df = df[df["provider"] == provider]
    if run_id is not None:
        df = df[df["run_id"] == run_id]
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


def accuracy_at_thresholds(df: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    """For each confidence threshold: coverage (fraction of all predictions clearing it),
    accuracy among those, and explicit correct/wrong counts.

    This is the "would thresholding on confidence actually work as a trust gate"
    question -- e.g. "if we only acted on confidence >= 0.90, what fraction of
    predictions would that cover, and how accurate would they be?" Still
    descriptive, not calibration: it doesn't fit or validate a model.
    """
    if thresholds is None:
        thresholds = DEFAULT_ACCURACY_THRESHOLDS
    n_total = len(df)
    rows = []
    for t in thresholds:
        subset = df[df["confidence"] >= t]
        n = len(subset)
        n_correct = int(subset["correct"].eq(True).sum())
        n_wrong = int(subset["correct"].eq(False).sum())
        rows.append(
            {
                "threshold": t,
                "coverage": (n / n_total) if n_total else None,
                "n": n,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "accuracy": (n_correct / n) if n else None,
            }
        )
    return pd.DataFrame(rows)


def confidence_histogram(df: pd.DataFrame, bins: list[float] | None = None) -> pd.DataFrame:
    """Count of correct vs. incorrect predictions in each confidence bin."""
    if bins is None:
        bins = DEFAULT_HISTOGRAM_BINS
    binned = pd.cut(df["confidence"], bins=bins, include_lowest=True).astype(str)
    counts = df.groupby([binned, "correct"], observed=True).size().unstack(fill_value=0)
    counts.index.name = "confidence_bin"
    return counts.reset_index()


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
    parser.add_argument("--run-id", type=str, default=None, help="Restrict to one exact run (default: all matching runs pooled together).")
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
    run_id: str | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist yet -- run an experiment (e.g. bitext_hard_choice.py) first to produce labeled results.")

    df = load_scored_results(input_path, experiment=experiment, provider=provider, run_id=run_id)
    if df.empty:
        raise ValueError("No scored rows (both 'correct' and 'confidence' populated) found for the given filters.")

    distribution = confidence_distribution_by_correctness(df)
    wrong_confident = confidently_wrong(df, threshold=high)
    low_conf = low_confidence_examples(df, threshold=low)
    quadrants = quadrant_examples(df, high=high, low=low)
    threshold_sweep = accuracy_at_thresholds(df)
    histogram = confidence_histogram(df)

    summary = {
        "n_total": len(df),
        "high_confidence_threshold": high,
        "low_confidence_threshold": low,
        "distribution_by_correctness": distribution.to_dict(orient="records"),
        "n_confidently_wrong": len(wrong_confident),
        "confidently_wrong_rate": len(wrong_confident) / len(df),
        "n_low_confidence": len(low_conf),
        "accuracy_at_thresholds": threshold_sweep.to_dict(orient="records"),
        "confidence_histogram": histogram.to_dict(orient="records"),
        "quadrant_examples": {name: rows.to_dict(orient="records") for name, rows in quadrants.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "confidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps({k: v for k, v in summary.items() if k not in ("quadrant_examples", "confidence_histogram", "accuracy_at_thresholds")}, indent=2, default=str))
    print("\naccuracy at confidence thresholds (coverage = fraction of all predictions clearing that threshold):")
    print(threshold_sweep.to_string(index=False))
    print("\nconfidence histogram (correct vs. wrong counts per bin):")
    print(histogram.to_string(index=False))
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
    run(args.input, args.experiment, args.provider, args.high_threshold, args.low_threshold, args.output_dir, run_id=args.run_id)


if __name__ == "__main__":
    main()
