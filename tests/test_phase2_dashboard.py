"""Headless tests of the Phase II dashboard tabs via Streamlit's AppTest, on synthetic artifacts.

The results root, baseline and stress directories are redirected to temp dirs with environment variables,
so nothing here touches real results and nothing calls a model.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from dataset_loaders.massive import LOCALES  # noqa: E402
from phase2.calibration import run_calibration  # noqa: E402
from phase2.failures import run_failures  # noqa: E402
from phase2.multilingual import run_multilingual  # noqa: E402
from phase2.selective import run_selective  # noqa: E402
from phase2_tabs import PHASE2_TABS, phase2_renderers  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
PHASE1_TABS = ["Overview", "Hard Choices", "Multilingual", "Choice Scaling", "Parallel Decisions", "Jev vs LLM", "Confidence", "Try It Yourself"]


def canonical() -> pd.DataFrame:
    rows = []

    def add(exp, prov, ex, loc, pred, conf, truth="a", dataset="bitext"):
        rows.append({"raw_row": len(rows), "run_id": f"{exp}-run", "experiment": exp, "provider": prov, "example_id": ex, "dataset": dataset, "locale": loc, "ground_truth": truth, "prediction": pred, "correct": pred == truth, "confidence": conf,
                     "candidates": ["a", "b", "c", "d", "e"], "text": f"message {ex}", "latency_ms": 100.0 if prov == "jev" else 900.0, "estimated_cost_usd": 0.001 if prov == "jev" else 0.01, "input_tokens": 50.0, "output_tokens": 5.0, "error": None, "request_id": "r"})

    for i in range(40):
        for prov in ("jev", "gpt-4o-mini"):
            add("bitext_hard_choice", prov, f"b{i}", None, "a" if i % 5 else "b", 0.55 + (i % 9) * 0.05)
    for i in range(6):
        for prov in ("jev", "gpt-4o-mini"):
            for loc in LOCALES:
                add("multilingual", prov, f"m{i}", loc, "b" if (i == 0 or (i == 1 and loc == "ja-JP")) else "a", 0.6 + 0.05 * i, dataset="massive")
    return pd.DataFrame(rows)


@pytest.fixture
def populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    baseline, root = tmp_path / "baseline", tmp_path / "phase2"
    baseline.mkdir()
    canonical().to_parquet(baseline / "canonical_results.parquet", index=False)
    run_calibration(baseline, root / "calibration", verify=False)
    run_selective(baseline, root / "selective_prediction", verify=False)
    run_multilingual(baseline, root / "multilingual", verify=False)
    run_failures(baseline, root, root / "failures", verify=False)
    monkeypatch.setenv("JEV_PHASE2_RESULTS_DIR", str(root))
    monkeypatch.setenv("JEV_BASELINE_DIR", str(baseline))
    monkeypatch.setenv("JEV_STRESS_DIR", str(tmp_path / "stress"))
    return root


@pytest.fixture
def empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("JEV_PHASE2_RESULTS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("JEV_BASELINE_DIR", str(tmp_path / "none_baseline"))
    monkeypatch.setenv("JEV_STRESS_DIR", str(tmp_path / "none_stress"))
    return tmp_path


def run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def text_of(at: AppTest) -> str:
    return " ".join([*(m.value for m in at.markdown), *(c.value for c in at.caption)])


def test_phase2_tab_registry_matches_the_renderers_and_the_spec() -> None:
    assert PHASE2_TABS == ["Calibration", "Risk / Coverage", "Stability", "Context Stress", "OOD", "Adversarial", "Multilingual Reliability", "Failure Museum", "Cascade Simulator"]
    assert len(phase2_renderers()) == len(PHASE2_TABS)


def test_phase1_tabs_are_preserved_in_place_and_phase2_tabs_are_appended(empty: Path) -> None:
    at = run_app()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels[:8] == PHASE1_TABS and labels[8:] == PHASE2_TABS  # Phase I indexes unchanged (Try It Yourself is still tabs[7])


def test_every_phase2_tab_degrades_gracefully_when_no_artifacts_exist(empty: Path) -> None:
    at = run_app()
    assert not at.exception and not at.error
    assert len(at.info) >= 9  # each Phase II tab shows a "no results recorded yet" message with the command to run
    info = " ".join(i.value for i in at.info)
    for command in ("run_calibration.py", "run_selective.py", "run_stability.py", "run_context_pollution.py", "run_ood.py", "run_adversarial.py", "run_multilingual.py", "run_failures.py"):
        assert command in info


def test_populated_dashboard_renders_calibration_and_shows_n_and_provider_labels(populated: Path) -> None:
    at = run_app()
    assert not at.exception
    body = text_of(at)
    assert "ECE" in body and "Brier (correctness)" in body and "binary, NOT multiclass" in body and "n (labeled predictions)" in body
    assert at.selectbox(key="p2_cal_group").value.startswith("Jev (primary)")  # Jev is the default group, clearly labelled
    assert any("reference LLM" in o for o in at.selectbox(key="p2_cal_group").options)  # the reference LLM is offered separately, labelled


def test_risk_coverage_slider_changes_the_reported_coverage(populated: Path) -> None:
    at = run_app()
    slider = at.slider(key="p2_rc_threshold")
    before = text_of(at)
    slider.set_value(0.55).run()
    low = text_of(at)
    slider.set_value(0.95).run()
    high = text_of(at)
    assert not at.exception and "Coverage (accepted / total)" in low
    assert low != high and before != high  # the KPIs are computed from the threshold, not static


def test_multilingual_reliability_shows_consistency_counts_from_the_files(populated: Path) -> None:
    at = run_app()
    body = text_of(at)
    assert "Correct in all locales" in body and "Wrong in all locales" in body
    assert "6 aligned source IDs" in body and "8 locales" in body  # counts come from the summary file, not hard-coded text
    assert "not ranked" in body


def test_failure_museum_filters_and_inspection(populated: Path) -> None:
    at = run_app()
    body = text_of(at)
    assert "failures match" in body
    at.multiselect(key="p2_fm_exp").set_value(["multilingual"]).run()
    assert not at.exception
    assert "failures match" in text_of(at)
    picked = at.selectbox(key="p2_fm_pick")
    assert picked.value.startswith("F-")
    at.multiselect(key="p2_fm_tags").set_value(["language_specific"]).run()
    assert not at.exception


def test_cascade_simulator_controls_change_stage_shares_and_are_labelled(populated: Path) -> None:
    at = run_app()
    body = text_of(at)
    assert "SIMULATED" in body and "NOT a measured human" in body and "ESTIMATED" in body
    assert "Handled by Jev" in body and "Handled by oracle (simulated)" in body
    at.slider(key="p2_cas_jev").set_value(0.50).run()
    all_jev = text_of(at)
    at.slider(key="p2_cas_jev").set_value(0.99).run()
    strict = text_of(at)
    assert all_jev != strict and not at.exception
    at.checkbox(key="p2_cas_llm").uncheck().run()
    at.checkbox(key="p2_cas_oracle").uncheck().run()
    assert not at.exception and "Unhandled" in text_of(at)  # oracle off + LLM off leaves requests unhandled instead of counting them right


def test_dashboard_makes_no_model_calls_and_does_not_need_a_key(populated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise AssertionError("the Phase II dashboard must not call a model")

    monkeypatch.setattr("clients.jev_client.AsyncJevClient", boom)
    monkeypatch.setattr("clients.jev_client.JevClient", boom)
    at = run_app()
    assert not at.exception
