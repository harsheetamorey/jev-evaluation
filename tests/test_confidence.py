"""Unit tests for the confidence-behavior analysis functions -- synthetic data.

No real Jev results have been recorded in this environment (no live API key
this session), so these tests build a synthetic results DataFrame shaped
like what `evaluation.recorder` would produce, rather than depending on an
actual run.
"""

from pathlib import Path

import pandas as pd
import pytest

from experiments.confidence import (
    accuracy_at_thresholds,
    confidence_distribution_by_correctness,
    confidence_histogram,
    confidently_wrong,
    load_scored_results,
    low_confidence_examples,
    quadrant_examples,
    run,
)


def _row(example_id, correct, confidence, experiment="bitext_hard_choice", provider="jev", text="hello", error=None, run_id="run-1"):
    return {
        "example_id": example_id,
        "experiment": experiment,
        "provider": provider,
        "dataset": "bitext",
        "locale": None,
        "text": text,
        "prediction": "x" if correct else "y",
        "ground_truth": "x",
        "correct": correct,
        "confidence": confidence,
        "error": error,
        "run_id": run_id,
    }


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("1", True, 0.95),
            _row("2", True, 0.90),
            _row("3", True, 0.40),
            _row("4", False, 0.92),  # confidently wrong
            _row("5", False, 0.30),
            _row("6", False, 0.20),
            _row("7", None, None, error="timeout"),  # errored, must be excluded by load_scored_results
        ]
    )


def test_load_scored_results_excludes_errored_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    _sample_df().to_parquet(path, index=False)

    df = load_scored_results(path)

    assert len(df) == 6
    assert df["correct"].notna().all()


def test_load_scored_results_filters_by_experiment_and_provider(tmp_path: Path) -> None:
    df_raw = _sample_df()
    df_raw.loc[0, "experiment"] = "multilingual"
    df_raw.loc[0, "provider"] = "gpt-4o-mini"
    path = tmp_path / "results.parquet"
    df_raw.to_parquet(path, index=False)

    filtered = load_scored_results(path, experiment="bitext_hard_choice")
    assert "1" not in set(filtered["example_id"])

    filtered_provider = load_scored_results(path, provider="jev")
    assert "1" not in set(filtered_provider["example_id"])


def test_confidence_distribution_shows_correct_rows_have_higher_mean() -> None:
    df = _sample_df().dropna(subset=["correct"])
    distribution = confidence_distribution_by_correctness(df)

    correct_row = distribution[distribution["correct"] == True].iloc[0]  # noqa: E712
    incorrect_row = distribution[distribution["correct"] == False].iloc[0]  # noqa: E712
    assert correct_row["n"] == 3
    assert incorrect_row["n"] == 3
    assert correct_row["mean"] == pytest.approx((0.95 + 0.90 + 0.40) / 3)
    assert incorrect_row["mean"] == pytest.approx((0.92 + 0.30 + 0.20) / 3)


def test_confidence_distribution_on_empty_df_returns_empty_frame() -> None:
    distribution = confidence_distribution_by_correctness(pd.DataFrame(columns=["correct", "confidence"]))
    assert distribution.empty


def test_confidently_wrong_finds_the_high_confidence_mistake() -> None:
    df = _sample_df().dropna(subset=["correct"])
    result = confidently_wrong(df, threshold=0.85)

    assert list(result["example_id"]) == ["4"]


def test_low_confidence_examples_ignores_correctness() -> None:
    df = _sample_df().dropna(subset=["correct"])
    result = low_confidence_examples(df, threshold=0.5)

    assert set(result["example_id"]) == {"3", "5", "6"}


def test_quadrant_examples_puts_each_row_in_exactly_one_bucket() -> None:
    df = _sample_df().dropna(subset=["correct"])
    quadrants = quadrant_examples(df, high=0.85, low=0.5, n=10)

    assert set(quadrants["high_confidence_correct"]["example_id"]) == {"1", "2"}
    assert set(quadrants["high_confidence_wrong"]["example_id"]) == {"4"}
    assert set(quadrants["low_confidence_correct"]["example_id"]) == {"3"}
    assert set(quadrants["low_confidence_wrong"]["example_id"]) == {"5", "6"}


def test_quadrant_examples_respects_n_cap_deterministically() -> None:
    rows = [_row(str(i), True, 0.95) for i in range(10)]
    df = pd.DataFrame(rows)

    first = quadrant_examples(df, high=0.85, low=0.5, n=3, seed=42)
    second = quadrant_examples(df, high=0.85, low=0.5, n=3, seed=42)

    assert len(first["high_confidence_correct"]) == 3
    assert list(first["high_confidence_correct"]["example_id"]) == list(second["high_confidence_correct"]["example_id"])


def test_run_raises_clearly_when_input_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist yet"):
        run(tmp_path / "nope.parquet", None, None, 0.85, 0.5, tmp_path)


def test_run_raises_clearly_when_no_scored_rows_match(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    pd.DataFrame([_row("1", None, None, error="boom")]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="No scored rows"):
        run(path, None, None, 0.85, 0.5, tmp_path)


def test_run_writes_summary_json(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    _sample_df().to_parquet(path, index=False)

    summary = run(path, None, None, 0.85, 0.5, tmp_path)

    assert summary["n_total"] == 6
    assert summary["n_confidently_wrong"] == 1
    assert (tmp_path / "confidence_summary.json").exists()


def test_load_scored_results_filters_by_run_id(tmp_path: Path) -> None:
    df_raw = _sample_df()
    df_raw.loc[0, "run_id"] = "other-run"
    path = tmp_path / "results.parquet"
    df_raw.to_parquet(path, index=False)

    filtered = load_scored_results(path, run_id="run-1")
    assert "1" not in set(filtered["example_id"])
    assert len(filtered) == 5  # 6 scored rows minus the one reassigned to "other-run"


def test_accuracy_at_thresholds_reports_coverage_and_accuracy() -> None:
    df = _sample_df().dropna(subset=["correct"])  # 3 correct (0.95, 0.90, 0.40), 3 wrong (0.92, 0.30, 0.20)
    result = accuracy_at_thresholds(df, thresholds=[0.5, 0.9])

    row_50 = result[result["threshold"] == 0.5].iloc[0]
    # >=0.5: correct 0.95, 0.90; wrong 0.92 -> 2 correct, 1 wrong, coverage 3/6
    assert row_50["n"] == 3
    assert row_50["n_correct"] == 2
    assert row_50["n_wrong"] == 1
    assert row_50["coverage"] == pytest.approx(0.5)
    assert row_50["accuracy"] == pytest.approx(2 / 3)

    row_90 = result[result["threshold"] == 0.9].iloc[0]
    # >=0.9: correct 0.95, 0.90; wrong 0.92 -> same 3 rows here too
    assert row_90["n"] == 3
    assert row_90["accuracy"] == pytest.approx(2 / 3)


def test_accuracy_at_thresholds_handles_empty_coverage() -> None:
    df = _sample_df().dropna(subset=["correct"])
    result = accuracy_at_thresholds(df, thresholds=[0.999])
    row = result.iloc[0]
    assert row["n"] == 0
    assert row["accuracy"] is None


def test_confidence_histogram_counts_correct_and_wrong_per_bin() -> None:
    df = _sample_df().dropna(subset=["correct"])
    result = confidence_histogram(df, bins=[0.0, 0.5, 1.0])

    # bin (0.0, 0.5]: correct=0.40 (1 correct), wrong=0.30,0.20 (2 wrong)
    # bin (0.5, 1.0]: correct=0.95,0.90 (2 correct), wrong=0.92 (1 wrong)
    totals = {False: result[False].sum(), True: result[True].sum()}
    assert totals[True] == 3
    assert totals[False] == 3
