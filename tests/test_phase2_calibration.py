"""Synthetic-data tests for confidence calibration. No network or API access."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase2.baseline import BaselineError
from phase2.calibration import (
    bin_index,
    brier_correctness,
    calibration_tables,
    dedup_measurements,
    eligible,
    expected_calibration_error,
    reliability_bins,
    run_calibration,
    summarize,
)


def preds(conf: list[float], correct: list[bool], **cols) -> pd.DataFrame:
    n = len(conf)
    base = {
        "raw_row": range(n),
        "run_id": "r1",
        "experiment": "exp",
        "provider": "jev",
        "dataset": "bitext",
        "example_id": [f"e{i}" for i in range(n)],
        "locale": None,
        "confidence": conf,
        "correct": correct,
        "candidates": [["a", "b"]] * n,
    }
    return pd.DataFrame({**base, **cols})


def ece_of(df: pd.DataFrame) -> float | None:
    return expected_calibration_error(reliability_bins(eligible(df)))


def test_perfectly_calibrated_has_zero_ece_and_gap() -> None:
    df = preds([0.8] * 10, [True] * 8 + [False] * 2)
    bins = reliability_bins(df)
    row = bins[bins["sample_count"] > 0].iloc[0]
    assert row["bin_label"] == "[0.8, 0.9)" and row["sample_count"] == 10
    assert row["calibration_gap"] == pytest.approx(0.0)
    assert ece_of(df) == pytest.approx(0.0)


def test_overconfident_has_positive_gap() -> None:
    df = preds([0.9] * 10, [True] * 5 + [False] * 5)
    row = reliability_bins(df).query("sample_count > 0").iloc[0]
    assert row["calibration_gap"] == pytest.approx(0.4)
    assert ece_of(df) == pytest.approx(0.4)


def test_underconfident_has_negative_gap_but_positive_ece() -> None:
    df = preds([0.3] * 10, [True] * 7 + [False] * 3)
    row = reliability_bins(df).query("sample_count > 0").iloc[0]
    assert row["calibration_gap"] == pytest.approx(-0.4)
    assert ece_of(df) == pytest.approx(0.4)  # ECE uses absolute gaps


def test_empty_bins_are_kept_with_zero_count_and_nan() -> None:
    bins = reliability_bins(preds([0.05, 0.95], [False, True]))
    assert len(bins) == 10
    empty = bins[bins["sample_count"] == 0]
    assert len(empty) == 8 and empty["mean_confidence"].isna().all() and empty["calibration_gap"].isna().all()


def test_confidence_exactly_zero_and_one_land_in_first_and_last_bin() -> None:
    assert bin_index(pd.Series([0.0, 0.1, 0.9, 0.99, 1.0])).tolist() == [0, 1, 9, 9, 9]
    bins = reliability_bins(preds([0.0, 1.0], [False, True]))
    assert bins.loc[0, "sample_count"] == 1 and bins.loc[9, "sample_count"] == 1
    assert bins.loc[9, "bin_label"] == "[0.9, 1.0]"


def test_bin_edges_are_robust_to_float_noise() -> None:
    assert bin_index(pd.Series([0.3, 0.6, 0.7, 0.29, 0.7000000001])).tolist() == [3, 6, 7, 2, 7]


def test_ece_weights_bins_by_sample_share() -> None:
    # 30 samples at 0.9 with accuracy 0.6 (gap 0.3); 10 samples at 0.1 with accuracy 0.1 (gap 0).
    df = preds([0.9] * 30 + [0.1] * 10, [True] * 18 + [False] * 12 + [True] + [False] * 9)
    assert ece_of(df) == pytest.approx(30 / 40 * 0.3 + 10 / 40 * 0.0)


def test_brier_known_examples() -> None:
    assert brier_correctness(preds([1.0, 0.0], [True, False])) == 0.0
    assert brier_correctness(preds([1.0, 0.0], [False, True])) == 1.0
    assert brier_correctness(preds([0.5] * 4, [True, False, True, False])) == pytest.approx(0.25)
    assert brier_correctness(preds([1.0, 0.0, 0.5, 0.8], [True, False, True, False])) == pytest.approx((0 + 0 + 0.25 + 0.64) / 4)
    assert brier_correctness(preds([], [])) is None


def test_summary_splits_confidence_by_correctness() -> None:
    s = summarize(preds([0.9, 0.7, 0.4], [True, True, False]))
    assert s["n"] == 3 and s["n_correct"] == 2 and s["n_incorrect"] == 1
    assert s["mean_confidence_correct"] == pytest.approx(0.8) and s["mean_confidence_incorrect"] == pytest.approx(0.4)


def test_eligible_drops_unlabeled_and_rejects_out_of_range() -> None:
    df = preds([0.9, np.nan, 0.5], [True, None, None])
    assert len(eligible(df)) == 1
    with pytest.raises(ValueError):
        eligible(preds([1.2], [True]))


def test_runs_are_not_pooled_at_per_run_level_and_pooling_is_labelled() -> None:
    df = pd.concat([preds([0.9] * 4, [True] * 4, run_id="r1", experiment="e1"), preds([0.9] * 4, [False] * 4, run_id="r2", experiment="e2")], ignore_index=True)
    df["raw_row"] = range(len(df))
    bins, summary = calibration_tables(df)
    per_run = summary[summary["level"] == "per_run"].set_index("run_id")
    assert per_run.loc["r1", "accuracy"] == 1.0 and per_run.loc["r2", "accuracy"] == 0.0
    pooled = summary[summary["level"] == "provider"].iloc[0]
    assert pooled["pooled_across_workloads"] and pooled["n"] == 8
    assert not per_run["pooled_across_workloads"].any()
    assert (bins.groupby(["level", "run_id", "experiment", "provider", "dataset", "num_choices"], dropna=False).size() == 10).all()  # every bin present for every group


def test_pooled_levels_count_remeasured_examples_once_using_latest_row() -> None:
    df = pd.concat([preds([0.9, 0.9], [False, False], run_id="old"), preds([0.9, 0.9], [True, True], run_id="new")], ignore_index=True)
    df["raw_row"] = range(len(df))
    assert len(dedup_measurements(df)) == 2
    _, summary = calibration_tables(df)
    assert summary[summary["level"] == "per_run"]["n"].sum() == 4
    pooled = summary[summary["level"] == "provider"].iloc[0]
    assert pooled["n"] == 2 and pooled["accuracy"] == 1.0  # latest rows, regardless of outcome


def test_run_calibration_writes_tables_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    preds([0.9, 0.2, 0.6], [True, False, None]).to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "calibration"
    meta = run_calibration(baseline, out, verify=False)
    assert (meta["canonical_rows"], meta["eligible_rows"]) == (3, 2)
    summary = pd.read_csv(out / "calibration_summary.csv")
    assert {"n", "ece", "brier_correctness", "brier_form"} <= set(summary.columns)
    assert len(pd.read_csv(out / "calibration_bins.csv")) == 10 * len(summary)
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_calibration(baseline, out, verify=False)
    assert not math.isnan(summary["ece"].iloc[0])
