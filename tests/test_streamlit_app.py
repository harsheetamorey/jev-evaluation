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
    choice_scaling_table,
    filter_experiment,
    filter_experiment_prefix,
    load_fanout_df,
    load_results_df,
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
