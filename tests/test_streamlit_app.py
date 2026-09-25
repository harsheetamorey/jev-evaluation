"""Tests for app/streamlit_app.py: its pure data-prep functions directly, and a
headless run of the actual app via Streamlit's AppTest harness (checking it
renders without exceptions in the realistic "no results recorded yet" state,
since no live Jev/LLM calls have been made in this environment).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from typesafe_sdk import TypeSafeError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from streamlit_app import (  # noqa: E402
    choice_scaling_insight,
    choice_scaling_table,
    confidence_gate_insight,
    fanout_insight,
    filter_experiment,
    filter_experiment_prefix,
    load_fanout_df,
    load_results_df,
    locale_spread_insight,
    mean_choices_by_group,
    metrics_table,
    overview_dashboard,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _bitext_row(provider="jev", correct=True, confidence=0.8, latency_ms=100.0):
    return {
        "experiment": "bitext_hard_choice",
        "provider": provider,
        "example_id": "1",
        "dataset": "bitext",
        "locale": None,
        "ground_truth": "billing",
        "prediction": "billing" if correct else "technical",
        "correct": correct,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "candidates": ["billing", "technical", "other"],
        "error": None,
        "input_tokens": 50,
        "output_tokens": 5,
        "estimated_cost_usd": 0.001,
        "text": "hello",
    }


# ---------------------------------------------------------------------------
# Pure data-prep functions
# ---------------------------------------------------------------------------


def test_load_results_df_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_results_df(tmp_path / "nope.parquet") is None


def test_load_fanout_df_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_fanout_df(tmp_path / "nope.parquet") is None


def test_load_results_df_returns_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    pd.DataFrame([_bitext_row()]).to_parquet(path, index=False)
    df = load_results_df(path)
    assert df is not None
    assert len(df) == 1


def test_filter_experiment() -> None:
    df = pd.DataFrame([_bitext_row(), {**_bitext_row(), "experiment": "multilingual"}])
    assert len(filter_experiment(df, "bitext_hard_choice")) == 1


def test_filter_experiment_prefix() -> None:
    df = pd.DataFrame(
        [
            {**_bitext_row(), "experiment": "choice_scaling_k5"},
            {**_bitext_row(), "experiment": "choice_scaling_k10"},
            {**_bitext_row(), "experiment": "bitext_hard_choice"},
        ]
    )
    assert len(filter_experiment_prefix(df, "choice_scaling")) == 2


def test_metrics_table_has_expected_columns() -> None:
    table = metrics_table({"jev": {"accuracy": 1.0, "p50_latency_ms": 100, "p95_latency_ms": 100, "error_rate": 0.0, "n": 1}})
    assert "Accuracy" in table.columns
    assert "n" in table.columns
    assert "# choices" not in table.columns  # omitted unless choices_by_group is passed


def test_metrics_table_includes_choices_column_when_given() -> None:
    table = metrics_table(
        {"jev": {"accuracy": 1.0, "p50_latency_ms": 100, "p95_latency_ms": 100, "error_rate": 0.0, "n": 1}},
        choices_by_group={"jev": 5.0},
    )
    assert table.loc["jev", "# choices"] == 5.0


def test_mean_choices_by_group() -> None:
    df = pd.DataFrame(
        [
            {**_bitext_row(provider="jev"), "candidates": ["a", "b", "c", "d", "e"]},
            {**_bitext_row(provider="gpt-4o-mini"), "candidates": list("abcdefghij")},
        ]
    )
    result = mean_choices_by_group(df, "provider")
    assert result == {"jev": 5.0, "gpt-4o-mini": 10.0}


def test_overview_dashboard_groups_by_experiment() -> None:
    df = pd.DataFrame([_bitext_row(), {**_bitext_row(), "experiment": "multilingual"}])
    table = overview_dashboard(df)
    assert set(table.index) == {"bitext_hard_choice", "multilingual"}


def test_choice_scaling_table_derives_num_choices_from_candidates_length() -> None:
    rows = [
        {**_bitext_row(), "candidates": ["a", "b", "c", "d", "e"]},
        {**_bitext_row(), "candidates": list("abcdefghij")},
    ]
    table = choice_scaling_table(pd.DataFrame(rows))
    assert list(table["num_choices"]) == [5, 10]


def test_choice_scaling_by_provider_does_not_pool_models() -> None:
    from streamlit_app import choice_scaling_by_provider

    rows = [
        {**_bitext_row(provider="jev"), "candidates": ["a", "b", "c", "d", "e"]},
        {**_bitext_row(provider="gpt-4o-mini"), "candidates": ["a", "b", "c", "d", "e"]},
        {**_bitext_row(provider="jev"), "candidates": list("abcdefghij")},
    ]
    table = choice_scaling_by_provider(pd.DataFrame(rows))
    assert set(table["provider"]) == {"jev", "gpt-4o-mini"}
    assert set(table["num_choices"]) == {5, 10}


# ---------------------------------------------------------------------------
# Data-derived insight sentences
#
# These render under each chart as prose, so a sign or direction error is
# invisible in the UI -- it just reads as a confident wrong claim. Worth
# asserting on the actual wording rather than only that a string came back.
# ---------------------------------------------------------------------------


def test_choice_scaling_insight_reports_a_decline_as_negative() -> None:
    """A 20-point drop must not print as "+20 pts", which reads as a gain."""
    tidy = pd.DataFrame(
        [
            {"num_choices": 5, "provider": "jev", "accuracy": 0.90, "mean_confidence": 0.95},
            {"num_choices": 60, "provider": "jev", "accuracy": 0.70, "mean_confidence": 0.85},
        ]
    )
    text = choice_scaling_insight(tidy)
    assert "-20 pts" in text
    assert "+20 pts" not in text
    assert "registering the added difficulty" in text  # confidence fell too


def test_choice_scaling_insight_flags_confidence_that_does_not_move() -> None:
    """Accuracy collapsing while confidence stays pinned is the dangerous shape."""
    tidy = pd.DataFrame(
        [
            {"num_choices": 5, "provider": "jev", "accuracy": 0.90, "mean_confidence": 0.99},
            {"num_choices": 60, "provider": "jev", "accuracy": 0.60, "mean_confidence": 0.99},
        ]
    )
    assert "without noticing" in choice_scaling_insight(tidy)


def test_choice_scaling_insight_needs_at_least_two_points() -> None:
    tidy = pd.DataFrame([{"num_choices": 5, "provider": "jev", "accuracy": 0.9, "mean_confidence": 0.9}])
    assert choice_scaling_insight(tidy) is None


def test_locale_spread_insight_names_best_and_worst_and_refuses_to_rank() -> None:
    rows = []
    for locale, n_correct in (("en-US", 90), ("ta-IN", 60)):
        for i in range(100):
            rows.append({**_bitext_row(correct=i < n_correct), "experiment": "multilingual", "locale": locale})
    text = locale_spread_insight(pd.DataFrame(rows))
    assert "en-US" in text and "ta-IN" in text
    assert "30-point spread" in text
    assert "not a ranking" in text or "not separable" in text


def test_locale_spread_insight_claims_separation_only_when_intervals_clear_each_other() -> None:
    """Both branches must be reachable.

    Separation is (worse locale's CI high < better locale's CI low). Comparing the
    outer bounds instead is never true, which would pin the sentence to the
    "overlapping" wording no matter how far apart the locales actually are.
    """
    def _spread(n_per_locale: int, top_rate: float, bottom_rate: float) -> str:
        rows = []
        for locale, rate in (("en-US", top_rate), ("ta-IN", bottom_rate)):
            n_correct = int(n_per_locale * rate)
            for i in range(n_per_locale):
                rows.append({**_bitext_row(correct=i < n_correct), "experiment": "multilingual", "locale": locale})
        return locale_spread_insight(pd.DataFrame(rows))

    # n=20, 80% vs 70%: Wilson gives [0.58, 0.92] and [0.48, 0.86] -- overlapping.
    assert "not a ranking" in _spread(20, 0.80, 0.70)
    # n=400, 95% vs 55%: [0.92, 0.97] and [0.50, 0.60] -- cleanly separated.
    assert "do not overlap at 95%" in _spread(400, 0.95, 0.55)


def test_locale_spread_insight_needs_two_locales() -> None:
    rows = [{**_bitext_row(), "experiment": "multilingual", "locale": "en-US"}]
    assert locale_spread_insight(pd.DataFrame(rows)) is None


def test_fanout_insight_separates_measured_total_from_derived_per_question() -> None:
    summary = pd.DataFrame(
        [
            {"num_questions": 1, "total_latency_ms": 200.0, "latency_per_question_ms": 200.0},
            {"num_questions": 50, "total_latency_ms": 250.0, "latency_per_question_ms": 5.0},
        ]
    )
    text = fanout_insight(summary)
    assert "40× cheaper" in text
    assert "not a separate result" in text  # the per-question collapse is arithmetic


def test_confidence_gate_insight_uses_the_tightest_qualifying_threshold() -> None:
    """0.85 is equidistant from 0.80 and 0.90; quoting 0.80 would overstate coverage."""
    gates = pd.DataFrame(
        [
            {"threshold": 0.80, "coverage": 0.90, "n": 900, "n_correct": 800, "n_wrong": 100, "accuracy": 0.889},
            {"threshold": 0.90, "coverage": 0.70, "n": 700, "n_correct": 680, "n_wrong": 20, "accuracy": 0.971},
        ]
    )
    text = confidence_gate_insight(gates)
    assert "0.90" in text
    assert "20 wrong predictions still clear the gate" in text


def test_confidence_gate_insight_handles_an_empty_sweep() -> None:
    assert confidence_gate_insight(pd.DataFrame(columns=["threshold", "coverage", "accuracy", "n_wrong"])) is None


# ---------------------------------------------------------------------------
# Headless full-app run (no results recorded -- the actual current repo state)
# ---------------------------------------------------------------------------


def test_app_runs_without_exceptions_with_no_data() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("tab_index", range(8))
def test_each_tab_selectable_without_exception(tab_index: int) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.tabs[tab_index].run()
    assert not at.exception


def _raise_no_key(*args, **kwargs):
    raise TypeSafeError("No API key was provided. Pass api_key or set the TYPESAFE_API_KEY environment variable.")


def test_try_it_yourself_classify_without_key_shows_clean_error_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates the no-key failure deterministically via a monkeypatched
    AsyncJevClient, rather than depending on whether this environment's .env
    happens to have a real key -- otherwise this test would make a real live
    Jev call (and fail its assertion) whenever a real key is configured, as
    happened once a real TYPESAFE_API_KEY was added during this session.

    Still exercises the real code path: the try/except in
    render_try_it_yourself around the live Jev call must catch this and
    display it, not crash the whole script.
    """
    monkeypatch.setattr("clients.jev_client.AsyncJevClient", _raise_no_key)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.tabs[7].text_area[0].input("I want to cancel my order").run()
    at.tabs[7].button[0].click().run()

    assert not at.exception
    error_texts = " ".join(el.value for el in at.tabs[7].error)
    assert "API key" in error_texts or "TYPESAFE_API_KEY" in error_texts


def test_app_runs_without_exceptions_with_real_data() -> None:
    """Exercise every "has data" branch (metrics tables, charts, cross-language
    consistency, choice-scaling grouping, fan-out aggregation) with a small
    synthetic multi-experiment dataset at the app's real default paths --
    since main() reads those paths unconditionally, not via an injectable
    argument. Guarded to skip rather than clobber if real results already
    exist, and always cleans up afterward.
    """
    from evaluation.recorder import DEFAULT_RESULTS_PATH

    fanout_path = DEFAULT_RESULTS_PATH.parent / "fanout_results.parquet"
    if DEFAULT_RESULTS_PATH.exists() or fanout_path.exists():
        pytest.skip("Real results already exist at the default path; not overwriting them for this test.")

    rows = [
        _bitext_row(provider="jev", correct=True, confidence=0.9),
        _bitext_row(provider="jev", correct=False, confidence=0.3),
        _bitext_row(provider="gpt-4o-mini", correct=True, confidence=0.8),
        {**_bitext_row(), "experiment": "multilingual", "locale": "en-US"},
        {**_bitext_row(), "experiment": "multilingual", "locale": "hi-IN", "correct": False},
        {**_bitext_row(), "experiment": "choice_scaling_k5", "candidates": list("abcde")},
        {**_bitext_row(), "experiment": "choice_scaling_k10", "candidates": list("abcdefghij")},
    ]
    fanout_rows = [
        {"message_id": "m1", "num_questions": 1, "total_latency_ms": 100.0, "latency_per_question_ms": 100.0, "input_chars": 10, "input_tokens": 50, "output_tokens": 5, "error": None},
        {"message_id": "m1", "num_questions": 5, "total_latency_ms": 300.0, "latency_per_question_ms": 60.0, "input_chars": 10, "input_tokens": 150, "output_tokens": 20, "error": None},
    ]

    DEFAULT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(DEFAULT_RESULTS_PATH, index=False)
    pd.DataFrame(fanout_rows).to_parquet(fanout_path, index=False)
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception
    finally:
        DEFAULT_RESULTS_PATH.unlink(missing_ok=True)
        fanout_path.unlink(missing_ok=True)
