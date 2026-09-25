"""Synthetic-data tests for selective prediction / risk-coverage. No network or API access."""

from pathlib import Path

import pandas as pd
import pytest

from phase2.baseline import BaselineError
from phase2.selective import (
    STANDARD_THRESHOLDS,
    THRESHOLDS,
    accuracy_is_non_decreasing,
    coverage_is_non_increasing,
    risk_coverage,
    risk_coverage_tables,
    run_selective,
)


def rows(conf: list[float], correct: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"confidence": conf, "correct": correct})


def at(curve: pd.DataFrame, t: float) -> pd.Series:
    return curve[curve["threshold"] == t].iloc[0]


def test_threshold_grid_is_0_50_to_0_99_in_hundredths_and_includes_standard_points() -> None:
    assert THRESHOLDS[0] == 0.5 and THRESHOLDS[-1] == 0.99 and len(THRESHOLDS) == 50
    assert all(round(b - a, 2) == 0.01 for a, b in zip(THRESHOLDS, THRESHOLDS[1:], strict=False))
    assert set(STANDARD_THRESHOLDS) <= set(THRESHOLDS)
    flagged = risk_coverage(rows([0.9], [True]))
    assert set(flagged.loc[flagged["is_standard_threshold"], "threshold"]) == set(STANDARD_THRESHOLDS)


def test_all_predictions_accepted() -> None:
    r = at(risk_coverage(rows([0.9, 0.8, 1.0], [True, False, True])), 0.5)
    assert (r["accepted"], r["rejected"], r["coverage"], r["rejected_fraction"]) == (3, 0, 1.0, 0.0)
    assert r["accepted_accuracy"] == pytest.approx(2 / 3) and r["risk"] == pytest.approx(1 / 3)


def test_no_predictions_accepted_gives_null_accuracy_and_risk() -> None:
    r = at(risk_coverage(rows([0.2, 0.3], [True, False])), 0.5)
    assert (r["accepted"], r["rejected"], r["coverage"]) == (0, 2, 0.0)
    assert r["accepted_accuracy"] is None and r["risk"] is None  # not 0, not invented
    assert r["rejected_accuracy"] == 0.5


def test_threshold_exactly_equal_to_confidence_is_accepted() -> None:
    curve = risk_coverage(rows([0.7, 0.7, 0.69], [True, True, False]))
    assert at(curve, 0.7)["accepted"] == 2
    assert at(curve, 0.71)["accepted"] == 0


def test_coverage_and_risk_are_computed_correctly() -> None:
    # 10 predictions: 4 below 0.8 (1 correct), 6 at/above (5 correct).
    df = rows([0.6, 0.6, 0.7, 0.75] + [0.8, 0.85, 0.9, 0.9, 0.95, 1.0], [True, False, False, False] + [True, True, True, True, True, False])
    r = at(risk_coverage(df), 0.8)
    assert (r["total"], r["accepted"], r["rejected"]) == (10, 6, 4)
    assert r["coverage"] == pytest.approx(0.6) and r["rejected_fraction"] == pytest.approx(0.4)
    assert r["accepted_accuracy"] == pytest.approx(5 / 6) and r["risk"] == pytest.approx(1 / 6)
    assert r["accepted_accuracy"] + r["risk"] == pytest.approx(1.0)
    assert (r["rejected_correct"], r["rejected_accuracy"]) == (1, 0.25)


def test_coverage_never_increases_with_threshold() -> None:
    curve = risk_coverage(rows([0.5, 0.55, 0.6, 0.7, 0.7, 0.85, 0.99, 1.0], [True] * 8))
    assert coverage_is_non_increasing(curve)
    assert curve["coverage"].is_monotonic_decreasing or curve["coverage"].nunique() > 1


def test_accepted_accuracy_may_be_non_monotonic() -> None:
    # Confident predictions are wrong more often than middling ones: accuracy falls as threshold rises.
    df = rows([0.6] * 4 + [0.95] * 4, [True] * 4 + [True, False, False, False])
    curve = risk_coverage(df)
    assert at(curve, 0.5)["accepted_accuracy"] == pytest.approx(5 / 8)
    assert at(curve, 0.9)["accepted_accuracy"] == pytest.approx(0.25)
    assert not accuracy_is_non_decreasing(curve)
    assert coverage_is_non_increasing(curve)


def test_empty_dataset() -> None:
    curve = risk_coverage(rows([], []))
    assert len(curve) == 50 and (curve["total"] == 0).all()
    assert curve["coverage"].isna().all() and curve["accepted_accuracy"].isna().all() and curve["risk"].isna().all()


def test_confidence_zero_and_one() -> None:
    curve = risk_coverage(rows([0.0, 1.0, 1.0], [False, True, True]))
    assert at(curve, 0.5)["accepted"] == 2 and at(curve, 0.99)["accepted"] == 2
    assert at(curve, 0.99)["accepted_accuracy"] == 1.0 and at(curve, 0.5)["rejected"] == 1


def _canonical(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_row": range(n),
            "run_id": "r1",
            "experiment": "multilingual",
            "provider": "jev",
            "dataset": "massive",
            "example_id": [f"e{i % 3}" for i in range(n)],
            "locale": ["en-US", "hi-IN"] * (n // 2),
            "confidence": [0.9, 0.6, 0.95, 0.4, 0.99, 0.7][:n],
            "correct": [True, False, True, None, True, False][:n],
            "candidates": [["a", "b"]] * n,
        }
    )


def test_groups_are_kept_separate_and_pooling_is_labelled() -> None:
    table = risk_coverage_tables(_canonical())
    per_run = table[(table["level"] == "per_run") & (table["threshold"] == 0.5)].iloc[0]
    assert per_run["total"] == 5 and not per_run["pooled_across_workloads"]  # unlabeled row excluded
    locales = table[(table["level"] == "per_run_locale") & (table["threshold"] == 0.5)].set_index("locale")
    assert set(locales.index) == {"en-US", "hi-IN"} and locales["total"].sum() == 5
    assert table[table["level"] == "provider_dataset"]["pooled_across_workloads"].all()


def test_run_selective_writes_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _canonical().to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    summary = run_selective(baseline, out, verify=False)
    assert summary["labeled_rows_analyzed"] == 5 and summary["no_best_threshold_selected"] is True
    csv = pd.read_csv(out / "risk_coverage.csv")
    assert {"total", "accepted", "coverage", "risk", "threshold"} <= set(csv.columns) and (csv["total"] >= 0).all()
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_selective(baseline, out, verify=False)
