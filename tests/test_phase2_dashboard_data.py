"""Tests for the Phase II dashboard's pure data layer. No Streamlit, no model calls."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.dashboard_data import (
    bins_for_group,
    dataset_info,
    filter_failures,
    group_label,
    group_options,
    load_csv,
    load_json,
    load_jsonl,
    load_matched,
    phase2_root,
    provider_label,
    risk_at,
    risk_curve,
    run_cascade_simulation,
    sparse_bin_warnings,
    workloads,
)


def test_root_is_redirectable_and_loaders_return_none_for_missing_or_empty_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_PHASE2_RESULTS_DIR", str(tmp_path))
    assert phase2_root() == tmp_path
    assert load_csv(tmp_path / "x.csv") is None and load_json(tmp_path / "x.json") is None and load_jsonl(tmp_path / "x.jsonl") is None
    (tmp_path / "e.csv").write_text("a,b\n")
    assert load_csv(tmp_path / "e.csv") is None
    (tmp_path / "ok.csv").write_text("a\n1\n")
    assert len(load_csv(tmp_path / "ok.csv")) == 1
    monkeypatch.setenv("JEV_STRESS_DIR", str(tmp_path))
    assert dataset_info("nothing") is None
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "dataset_manifest.json").write_text(json.dumps({"n_rows": 3}))
    assert dataset_info("ds") == {"n_rows": 3}


def test_jsonl_loader_is_strict_and_rejects_bare_nan(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('{"a": 1, "locale": null}\n{"a": 2, "locale": "en-US"}\n')
    assert len(load_jsonl(good)) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": NaN}\n')
    with pytest.raises(ValueError, match="non-strict JSON"):
        load_jsonl(bad)


def test_provider_labels_always_distinguish_jev_from_the_reference_llm() -> None:
    assert provider_label("jev") == "Jev (primary)" and provider_label("gpt-4o-mini") == "gpt-4o-mini (reference LLM)"


def summary_and_bins() -> tuple[pd.DataFrame, pd.DataFrame]:
    def summ(provider, run=None, level="provider_dataset"):
        return {"level": level, "pooled_across_workloads": level == "provider_dataset", "run_id": run, "experiment": None, "provider": provider, "dataset": "bitext", "num_choices": None, "n": 10, "ece": 0.1}

    summary = pd.DataFrame([summ("gpt-4o-mini"), summ("jev"), summ("jev", run="r1", level="per_run")])
    rows = []
    for _, s in summary.iterrows():
        for b in range(10):
            rows.append({**{c: s[c] for c in ["level", "run_id", "experiment", "provider", "dataset", "num_choices"]}, "bin_index": b, "bin_label": f"b{b}", "sample_count": 40 if b == 9 else (5 if b == 0 else 0)})
    return summary, pd.DataFrame(rows)


def test_group_options_lists_jev_first_and_labels_are_readable() -> None:
    summary, _ = summary_and_bins()
    labels = list(group_options(summary, "provider_dataset"))
    assert labels[0].startswith("Jev (primary)") and labels[1].startswith("gpt-4o-mini (reference LLM)")
    assert "dataset=bitext" in labels[0] and "run_id=r1" in group_label(summary.iloc[2])


def test_bins_are_matched_to_exactly_one_group_and_sparse_bins_are_flagged() -> None:
    summary, bins = summary_and_bins()
    jev_pooled = summary.iloc[1]
    gb = bins_for_group(bins, jev_pooled)
    assert len(gb) == 10 and (gb["provider"] == "jev").all() and gb["run_id"].isna().all()  # not confused with the per-run jev group
    assert len(bins_for_group(bins, summary.iloc[2])) == 10
    assert sparse_bin_warnings(gb) == ["b0: n=5"]  # the empty bins and the n=40 bin are not warned about
    assert sparse_bin_warnings(gb, min_n=50) == ["b0: n=5", "b9: n=40"]


def risk_table() -> pd.DataFrame:
    rows = []
    for t, acc, cov in [(0.5, 0.8, 1.0), (0.6, 0.85, 0.7), (0.7, None, 0.0)]:
        rows.append({"level": "provider", "run_id": None, "experiment": None, "provider": "jev", "dataset": None, "num_choices": None, "locale": None, "threshold": t, "total": 10, "accepted": round(cov * 10), "rejected": 10 - round(cov * 10),
                     "coverage": cov, "accepted_accuracy": acc, "risk": None if acc is None else 1 - acc})
    rows.append({**rows[0], "provider": "gpt-4o-mini"})
    return pd.DataFrame(rows)


def test_risk_curve_selection_and_null_accuracy_when_nothing_is_accepted() -> None:
    t = risk_table()
    curve = risk_curve(t, {"provider": "jev"}, "provider")
    assert curve["threshold"].tolist() == [0.5, 0.6, 0.7] and (curve["provider"] == "jev").all()
    r = risk_at(curve, 0.6)
    assert (r["accepted"], r["rejected"], r["coverage"]) == (7, 3, 0.7) and r["accepted_accuracy"] == pytest.approx(0.85) and r["risk"] == pytest.approx(0.15)
    empty = risk_at(curve, 0.7)
    assert empty["accepted"] == 0 and empty["accepted_accuracy"] is None and empty["risk"] is None  # not invented as 0
    with pytest.raises(ValueError, match="not on this curve"):
        risk_at(curve, 0.55)


def failures() -> pd.DataFrame:
    return pd.DataFrame([
        {"failure_id": "F1", "experiment": "multilingual", "provider": "jev", "locale": "ja-JP", "failure_tags": ["language_specific"], "confidence": 0.8, "ground_truth": "a", "prediction": "b"},
        {"failure_id": "F2", "experiment": "multilingual", "provider": "jev", "locale": "en-US", "failure_tags": ["high_confidence_wrong", "language_specific"], "confidence": 0.95, "ground_truth": "a", "prediction": "c"},
        {"failure_id": "F3", "experiment": "bitext_hard_choice", "provider": "gpt-4o-mini", "locale": None, "failure_tags": [], "confidence": None, "ground_truth": "b", "prediction": "a"},
    ])


def test_failure_filters_combine_with_and_and_tags_match_any() -> None:
    df = failures()
    assert len(filter_failures(df)) == 3
    assert set(filter_failures(df, experiments=["multilingual"])["failure_id"]) == {"F1", "F2"}
    assert set(filter_failures(df, tags=["high_confidence_wrong"])["failure_id"]) == {"F2"}
    assert set(filter_failures(df, tags=["language_specific", "high_confidence_wrong"])["failure_id"]) == {"F1", "F2"}
    assert set(filter_failures(df, locales=["ja-JP"], experiments=["multilingual"])["failure_id"]) == {"F1"}
    assert set(filter_failures(df, conf_range=(0.9, 1.0))["failure_id"]) == {"F2"}
    assert set(filter_failures(df, ground_truth=["b"])["failure_id"]) == {"F3"} and set(filter_failures(df, prediction=["c"])["failure_id"]) == {"F2"}
    assert set(filter_failures(df, providers=["gpt-4o-mini"])["failure_id"]) == {"F3"} and filter_failures(df, tags=["adversarial_manipulation"]).empty


def matched() -> pd.DataFrame:
    rows = [("a", 0.95, True, 100.0, 0.001, "a", 0.8, True, 900.0, 0.01), ("b", 0.90, False, 110.0, 0.001, "a", 0.8, True, 800.0, 0.01), ("b", 0.60, False, 120.0, 0.001, "a", 0.8, True, 700.0, 0.01), ("a", 0.50, True, 130.0, 0.001, "b", 0.8, False, 600.0, 0.01)]
    return pd.DataFrame([{"experiment": "exp", "example_id": f"e{i}", "locale": None, "num_choices": 5, "dataset": "bitext", "ground_truth": "a", "text": "hi", "jev_prediction": jp, "jev_confidence": jc, "jev_correct": jok, "jev_latency_ms": jl, "jev_estimated_cost_usd": jcost,
                         "llm_prediction": lp, "llm_confidence": lc, "llm_correct": lok, "llm_latency_ms": ll, "llm_estimated_cost_usd": lcost} for i, (jp, jc, jok, jl, jcost, lp, lc, lok, ll, lcost) in enumerate(rows)])


def test_cascade_simulation_responds_to_every_control_and_is_labelled_simulated() -> None:
    m = matched()
    assert workloads(m) == ["exp|k=5"]
    base = run_cascade_simulation(m, "exp|k=5", 0.90, True, True, llm_threshold=0.5)
    assert base["simulated"] is True and base["n_total"] == 4 and (base["n_jev"], base["n_llm"], base["n_oracle"]) == (2, 2, 0)
    assert base["overall_accuracy_including_oracle"] == 0.5 and base["coverage_rules"] == 0.0
    low = run_cascade_simulation(m, "exp|k=5", 0.0, True, True)
    assert low["n_jev"] == 4  # raising/lowering the Jev threshold moves work between stages
    no_llm = run_cascade_simulation(m, "exp|k=5", 0.90, False, True)
    assert (no_llm["n_llm"], no_llm["n_oracle"]) == (0, 2)  # LLM fallback switched off -> those requests reach the oracle
    no_oracle = run_cascade_simulation(m, "exp|k=5", 0.90, False, False)
    assert (no_oracle["n_oracle"], no_oracle["n_unhandled"]) == (0, 2)  # oracle off -> unresolved, not counted right or wrong
    strict_llm = run_cascade_simulation(m, "exp|k=5", 0.90, True, True, llm_threshold=0.9)
    assert strict_llm["n_llm"] == 0 and strict_llm["n_oracle"] == 2  # unvalidated LLM outputs go to the oracle
    with pytest.raises(ValueError, match="unknown workload"):
        run_cascade_simulation(m, "nope|k=5", 0.9, True, True)


def test_load_matched_reads_the_frozen_canonical_baseline(tmp_path: Path) -> None:
    assert load_matched(tmp_path) is None
    rows = []
    for i in range(3):
        for prov in ("jev", "gpt-4o-mini"):
            rows.append({"raw_row": len(rows), "run_id": "r", "experiment": "e", "provider": prov, "example_id": str(i), "dataset": "bitext", "locale": None, "ground_truth": "a", "candidates": ["a", "b"], "prediction": "a", "correct": True, "confidence": 0.9,
                         "latency_ms": 50.0, "estimated_cost_usd": 0.001, "text": "hi"})
    pd.DataFrame(rows).to_parquet(tmp_path / "canonical_results.parquet", index=False)
    m = load_matched(tmp_path)
    assert len(m) == 3 and {"jev_prediction", "llm_prediction", "text"} <= set(m.columns)
