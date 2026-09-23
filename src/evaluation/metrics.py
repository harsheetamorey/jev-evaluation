"""Summary metrics computed from a results DataFrame (see evaluation.recorder)."""

from typing import Any

import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute accuracy, latency percentiles, error rate, and confidence breakdowns.

    `correct` is `None` for errored predictions, so accuracy/confidence are computed
    only over scored (non-errored) rows; `error_rate` and latency cover every row.
    """
    total = len(df)
    if total == 0:
        return {
            "n": 0,
            "accuracy": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "error_rate": None,
            "confidence_when_correct": None,
            "confidence_when_incorrect": None,
        }

    scored = df["correct"].notna()
    correct_mask = df["correct"].eq(True)
    incorrect_mask = df["correct"].eq(False)

    def _mean_or_none(series: pd.Series) -> float | None:
        values = series.dropna()
        return float(values.mean()) if not values.empty else None

    return {
        "n": total,
        "accuracy": float(df.loc[scored, "correct"].astype(bool).mean()) if scored.any() else None,
        "p50_latency_ms": float(df["latency_ms"].quantile(0.5)),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "error_rate": float(df["error"].notna().mean()),
        "confidence_when_correct": _mean_or_none(df.loc[correct_mask, "confidence"]),
        "confidence_when_incorrect": _mean_or_none(df.loc[incorrect_mask, "confidence"]),
    }
