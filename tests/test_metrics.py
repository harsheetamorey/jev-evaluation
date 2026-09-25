"""Unit tests for evaluation.metrics.compute_metrics."""

import pandas as pd
import pytest

from evaluation.metrics import compute_cross_language_consistency, compute_metrics, compute_metrics_by_group, wilson_ci


def _row(correct, confidence, latency_ms, error=None):
    return {"correct": correct, "confidence": confidence, "latency_ms": latency_ms, "error": error}


def test_empty_dataframe_returns_all_none() -> None:
    df = pd.DataFrame(columns=["correct", "confidence", "latency_ms", "error"])
    metrics = compute_metrics(df)
    assert metrics == {
        "n": 0,
        "accuracy": None,
        "accuracy_ci_low": None,
        "accuracy_ci_high": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "error_rate": None,
        "confidence_when_correct": None,
        "confidence_when_incorrect": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_estimated_cost_usd": None,
    }


def test_accuracy_ignores_errored_rows() -> None:
    df = pd.DataFrame(
        [
            _row(True, 0.9, 100),
            _row(False, 0.6, 100),
            _row(None, None, 50, error="timeout"),
        ]
    )
    metrics = compute_metrics(df)
    assert metrics["n"] == 3
    assert metrics["accuracy"] == 0.5
    assert metrics["error_rate"] == pytest.approx(1 / 3)


def test_confidence_split_by_correctness() -> None:
    df = pd.DataFrame(
        [
            _row(True, 0.9, 10),
            _row(True, 0.7, 10),
            _row(False, 0.4, 10),
        ]
    )
    metrics = compute_metrics(df)
    assert metrics["confidence_when_correct"] == pytest.approx(0.8)
    assert metrics["confidence_when_incorrect"] == pytest.approx(0.4)


def test_latency_percentiles() -> None:
    df = pd.DataFrame([_row(True, 0.9, ms) for ms in [10, 20, 30, 40, 100]])
    metrics = compute_metrics(df)
    assert metrics["p50_latency_ms"] == 30
    assert metrics["p95_latency_ms"] == pytest.approx(88.0)


def test_all_errored_gives_none_accuracy_and_confidence() -> None:
    df = pd.DataFrame([_row(None, None, 50, error="boom"), _row(None, None, 60, error="boom")])
    metrics = compute_metrics(df)
    assert metrics["accuracy"] is None
    assert metrics["confidence_when_correct"] is None
    assert metrics["confidence_when_incorrect"] is None
    assert metrics["error_rate"] == 1.0


def test_token_and_cost_totals_are_summed() -> None:
    df = pd.DataFrame(
        [
            {**_row(True, 0.9, 10), "input_tokens": 100, "output_tokens": 10, "estimated_cost_usd": 0.01},
            {**_row(False, 0.4, 10), "input_tokens": 200, "output_tokens": 20, "estimated_cost_usd": 0.02},
        ]
    )
    metrics = compute_metrics(df)
    assert metrics["total_input_tokens"] == 300
    assert metrics["total_output_tokens"] == 30
    assert metrics["total_estimated_cost_usd"] == pytest.approx(0.03)


def test_token_and_cost_totals_are_none_when_columns_absent() -> None:
    df = pd.DataFrame([_row(True, 0.9, 10)])
    metrics = compute_metrics(df)
    assert metrics["total_input_tokens"] is None
    assert metrics["total_output_tokens"] is None
    assert metrics["total_estimated_cost_usd"] is None


def _multilingual_row(example_id, locale, correct, confidence=0.9, latency_ms=10, error=None):
    return {
        "example_id": example_id,
        "locale": locale,
        "correct": correct,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "error": error,
    }


def test_compute_metrics_by_group_splits_correctly() -> None:
    df = pd.DataFrame(
        [
            _multilingual_row("1", "en-US", True),
            _multilingual_row("2", "en-US", False),
            _multilingual_row("1", "hi-IN", True),
            _multilingual_row("2", "hi-IN", True),
        ]
    )
    by_locale = compute_metrics_by_group(df, "locale")
    assert set(by_locale) == {"en-US", "hi-IN"}
    assert by_locale["en-US"]["accuracy"] == 0.5
    assert by_locale["hi-IN"]["accuracy"] == 1.0
    assert by_locale["en-US"]["n"] == 2


def test_cross_language_consistency_histogram() -> None:
    df = pd.DataFrame(
        [
            # id "1": correct in both locales -> 2/2
            _multilingual_row("1", "en-US", True),
            _multilingual_row("1", "hi-IN", True),
            # id "2": correct in one of two locales -> 1/2
            _multilingual_row("2", "en-US", True),
            _multilingual_row("2", "hi-IN", False),
            # id "3": errored in one locale, counts as not-agreeing -> 1/2
            _multilingual_row("3", "en-US", True),
            _multilingual_row("3", "hi-IN", None, confidence=None, error="boom"),
        ]
    )
    result = compute_cross_language_consistency(df)

    breakdown_by_id = {entry["id"]: entry for entry in result["per_id"]}
    assert breakdown_by_id["1"] == {"id": "1", "correct_locales": 2, "total_locales": 2}
    assert breakdown_by_id["2"] == {"id": "2", "correct_locales": 1, "total_locales": 2}
    assert breakdown_by_id["3"] == {"id": "3", "correct_locales": 1, "total_locales": 2}
    assert result["histogram"] == {"2/2": 1, "1/2": 2}


def test_wilson_ci_matches_known_values() -> None:
    # Textbook check: 8/10 successes, 95% Wilson CI is approximately (0.492, 0.943).
    low, high = wilson_ci(8, 10)
    assert low == pytest.approx(0.492, abs=0.01)
    assert high == pytest.approx(0.943, abs=0.01)


def test_wilson_ci_is_symmetric_around_half_at_p_half() -> None:
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(1 - high, abs=1e-9)


def test_wilson_ci_narrows_with_larger_n_at_same_proportion() -> None:
    low_small, high_small = wilson_ci(80, 100)
    low_large, high_large = wilson_ci(800, 1000)
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_ci_returns_none_for_zero_n() -> None:
    assert wilson_ci(0, 0) == (None, None)


def test_compute_metrics_includes_accuracy_confidence_interval() -> None:
    df = pd.DataFrame([_row(True, 0.9, 10)] * 8 + [_row(False, 0.4, 10)] * 2)
    metrics = compute_metrics(df)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["accuracy_ci_low"] < metrics["accuracy"] < metrics["accuracy_ci_high"]
    assert metrics["accuracy_ci_low"] == pytest.approx(wilson_ci(8, 10)[0])
