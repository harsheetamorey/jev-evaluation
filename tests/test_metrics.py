"""Unit tests for evaluation.metrics.compute_metrics."""

import pandas as pd
import pytest

from evaluation.metrics import compute_metrics


def _row(correct, confidence, latency_ms, error=None):
    return {"correct": correct, "confidence": confidence, "latency_ms": latency_ms, "error": error}


def test_empty_dataframe_returns_all_none() -> None:
    df = pd.DataFrame(columns=["correct", "confidence", "latency_ms", "error"])
    metrics = compute_metrics(df)
    assert metrics == {
        "n": 0,
        "accuracy": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "error_rate": None,
        "confidence_when_correct": None,
        "confidence_when_incorrect": None,
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
