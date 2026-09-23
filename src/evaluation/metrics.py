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
            "total_input_tokens": None,
            "total_output_tokens": None,
            "total_estimated_cost_usd": None,
        }

    scored = df["correct"].notna()
    correct_mask = df["correct"].eq(True)
    incorrect_mask = df["correct"].eq(False)

    def _mean_or_none(series: pd.Series) -> float | None:
        values = series.dropna()
        return float(values.mean()) if not values.empty else None

    def _sum_or_none(col: str) -> float | None:
        if col not in df:
            return None
        values = df[col].dropna()
        return float(values.sum()) if not values.empty else None

    return {
        "n": total,
        "accuracy": float(df.loc[scored, "correct"].astype(bool).mean()) if scored.any() else None,
        "p50_latency_ms": float(df["latency_ms"].quantile(0.5)),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "error_rate": float(df["error"].notna().mean()),
        "confidence_when_correct": _mean_or_none(df.loc[correct_mask, "confidence"]),
        "confidence_when_incorrect": _mean_or_none(df.loc[incorrect_mask, "confidence"]),
        "total_input_tokens": _sum_or_none("input_tokens"),
        "total_output_tokens": _sum_or_none("output_tokens"),
        "total_estimated_cost_usd": _sum_or_none("estimated_cost_usd"),
    }


def compute_metrics_by_group(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, Any]]:
    """`compute_metrics()` applied separately to each value of `group_col` (e.g. "locale")."""
    return {str(key): compute_metrics(group_df) for key, group_df in df.groupby(group_col, sort=True)}


_COMPARISON_ROWS = [
    ("accuracy", "accuracy"),
    ("p50 latency (ms)", "p50_latency_ms"),
    ("p95 latency (ms)", "p95_latency_ms"),
    ("error rate", "error_rate"),
    ("input tokens", "total_input_tokens"),
    ("output tokens", "total_output_tokens"),
    ("estimated cost (usd)", "total_estimated_cost_usd"),
]


def format_comparison_table(by_provider: dict[str, dict[str, Any]]) -> str:
    """Render a side-by-side comparison table from compute_metrics_by_group(df, "provider").

    Column headers are each provider's exact label (e.g. "jev", "gpt-4o-mini") --
    never a generic "LLM" column.
    """
    providers = list(by_provider)
    col_width = max((len(p) for p in providers), default=0) + 2
    label_width = max(len(label) for label, _ in _COMPARISON_ROWS) + 2

    def _cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "-" if value is None else str(value)

    lines = ["metric".ljust(label_width) + "".join(p.rjust(col_width) for p in providers)]
    for label, key in _COMPARISON_ROWS:
        row = label.ljust(label_width) + "".join(_cell(by_provider[p].get(key)).rjust(col_width) for p in providers)
        lines.append(row)
    return "\n".join(lines)


def compute_cross_language_consistency(
    df: pd.DataFrame,
    id_col: str = "example_id",
    locale_col: str = "locale",
    correct_col: str = "correct",
) -> dict[str, Any]:
    """For each aligned ID, count how many locale predictions matched ground truth.

    An errored prediction (correct is None) counts as not-agreeing. Returns a per-ID
    breakdown and a histogram of agreement counts, e.g. how many aligned utterances
    had 8/8, 7/8, ... locales predict the correct intent.
    """
    correct_counts = df.groupby(id_col)[correct_col].apply(lambda s: int(s.fillna(False).sum()))
    locale_counts = df.groupby(id_col)[locale_col].nunique()

    breakdown = [
        {"id": str(example_id), "correct_locales": int(correct_counts[example_id]), "total_locales": int(locale_counts[example_id])}
        for example_id in correct_counts.index
    ]
    histogram: dict[str, int] = {}
    for entry in breakdown:
        key = f"{entry['correct_locales']}/{entry['total_locales']}"
        histogram[key] = histogram.get(key, 0) + 1

    return {"per_id": breakdown, "histogram": histogram}
