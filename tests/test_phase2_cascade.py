"""Tests for the offline Jev -> LLM cascade simulation. No model calls, no network."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase2.baseline import BaselineError
from phase2.cascade import (
    build_matched,
    estimate_missing_llm,
    reference_rows,
    run_cascade,
    simulate,
    summarize_threshold,
    threshold_table,
)


def matched(rows: list[tuple]) -> pd.DataFrame:
    """rows: (jev_pred, jev_conf, jev_ok, jev_lat, jev_cost, llm_pred, llm_ok, llm_lat, llm_cost)"""
    data = []
    for i, (jp, jc, jok, jl, jcost, lp, lok, ll, lcost) in enumerate(rows):
        data.append({"experiment": "exp", "example_id": f"e{i}", "locale": None, "num_choices": 5, "dataset": "bitext", "ground_truth": "a", "jev_prediction": jp, "jev_confidence": jc, "jev_correct": jok, "jev_latency_ms": jl, "jev_estimated_cost_usd": jcost,
                     "llm_prediction": lp, "llm_confidence": 0.5, "llm_correct": lok, "llm_latency_ms": ll, "llm_estimated_cost_usd": lcost})
    return pd.DataFrame(data)


ROWS = [
    ("a", 0.95, True, 100.0, 0.001, "a", True, 900.0, 0.010),  # confident, correct -> Jev
    ("b", 0.90, False, 110.0, 0.001, "a", True, 800.0, 0.010),  # exactly at 0.90 -> Jev (wrong)
    ("b", 0.60, False, 120.0, 0.001, "a", True, 700.0, 0.010),  # unsure -> LLM (right)
    ("a", 0.50, True, 130.0, 0.001, "b", False, 600.0, 0.010),  # unsure -> LLM (wrong)
]


def test_threshold_routing_is_inclusive_and_selects_the_right_final_prediction() -> None:
    sim = simulate(matched(ROWS), 0.90)
    assert sim["route"].tolist() == ["jev", "jev", "llm", "llm"]  # confidence == threshold is accepted
    assert sim["final_prediction"].tolist() == ["a", "b", "a", "b"]  # Jev's answer if accepted, else the LLM's
    assert [bool(x) for x in sim["final_correct"]] == [True, False, True, False]
    assert simulate(matched(ROWS), 0.91)["route"].tolist() == ["jev", "llm", "llm", "llm"]


def test_cascade_accuracy_stage_coverage_and_stage_errors() -> None:
    s = summarize_threshold(simulate(matched(ROWS), 0.90))
    assert (s["n_total"], s["n_handled_by_jev"], s["n_fell_back_to_llm"]) == (4, 2, 2)
    assert s["pct_handled_by_jev"] == 0.5 and s["pct_fell_back_to_llm"] == 0.5
    assert s["final_accuracy"] == 0.5 and s["n_final_correct"] == 2
    assert (s["jev_stage_errors"], s["llm_fallback_errors"]) == (1, 1)
    assert s["jev_stage_accuracy"] == 0.5 and s["llm_fallback_accuracy"] == 0.5
    assert s["n_handled_by_jev"] + s["n_fell_back_to_llm"] + s["n_unresolved_missing_llm"] == s["n_total"]


def test_all_or_nothing_thresholds() -> None:
    everyone_jev = summarize_threshold(simulate(matched(ROWS), 0.0))
    assert everyone_jev["pct_handled_by_jev"] == 1.0 and everyone_jev["n_fell_back_to_llm"] == 0 and everyone_jev["final_accuracy"] == 0.5
    everyone_llm = summarize_threshold(simulate(matched(ROWS), 1.01))
    assert everyone_llm["pct_fell_back_to_llm"] == 1.0 and everyone_llm["final_accuracy"] == 0.75 and everyone_llm["jev_stage_accuracy"] is None  # LLM is right on 3 of 4


def test_latency_is_sequential_on_fallback_and_percentiles_use_path_latencies() -> None:
    sim = simulate(matched(ROWS), 0.90)
    assert sim["path_latency_ms"].tolist() == [100.0, 110.0, 120.0 + 700.0, 130.0 + 600.0]  # accepted: Jev only; fallback: Jev + LLM
    s = summarize_threshold(sim)
    assert s["path_latency_p50_ms"] == pytest.approx(np.quantile([100, 110, 820, 730], 0.5))
    assert s["path_latency_p95_ms"] == pytest.approx(np.quantile([100, 110, 820, 730], 0.95))
    medians_average = (np.median([100, 110, 120, 130]) + np.median([900, 800, 700, 600])) / 2
    assert s["path_latency_p50_ms"] != pytest.approx(medians_average)  # not an average of provider medians


def test_cost_runs_jev_on_every_request_and_the_llm_only_on_fallbacks() -> None:
    sim = simulate(matched(ROWS), 0.90)
    assert sim["jev_cost_usd"].tolist() == [0.001] * 4 and sim["llm_cost_usd"].tolist() == [0.0, 0.0, 0.010, 0.010]
    s = summarize_threshold(sim)
    assert s["total_cost_usd_estimated"] == pytest.approx(4 * 0.001 + 2 * 0.010)
    assert s["cost_per_1000_requests_usd_estimated"] == pytest.approx((4 * 0.001 + 2 * 0.010) / 4 * 1000) and s["n_missing_cost"] == 0


def test_missing_cost_makes_the_total_unavailable_instead_of_zero() -> None:
    rows = [(*r[:4], None if i == 0 else r[4], *r[5:]) for i, r in enumerate(ROWS)]
    s = summarize_threshold(simulate(matched(rows), 0.90))
    assert s["total_cost_usd_estimated"] is None and s["cost_per_1000_requests_usd_estimated"] is None and s["n_missing_cost"] == 1
    fallback_missing = [(*r[:8], None if i == 2 else r[8]) for i, r in enumerate(ROWS)]  # LLM cost missing on a fallback row
    assert summarize_threshold(simulate(matched(fallback_missing), 0.90))["total_cost_usd_estimated"] is None


def test_fallback_without_an_llm_prediction_is_unresolved_not_counted_right_or_wrong() -> None:
    rows = [*ROWS[:2], ("b", 0.60, False, 120.0, 0.001, None, None, None, None), ROWS[3]]
    sim = simulate(matched(rows), 0.90)
    assert sim["route"].tolist() == ["jev", "jev", "unresolved_missing_llm", "llm"]
    s = summarize_threshold(sim)
    assert s["n_unresolved_missing_llm"] == 1 and s["n_fell_back_to_llm"] == 1
    assert s["final_accuracy"] == pytest.approx(1 / 3)  # accuracy over the 3 resolved examples only; the denominator is visible via n
    assert sim["path_latency_ms"].iloc[2] == 120.0 and sim["llm_cost_usd"].iloc[2] == 0.0  # no LLM latency/cost invented


def test_per_experiment_grouping_and_same_population_references() -> None:
    m = pd.concat([matched(ROWS), matched(ROWS).assign(experiment="other", num_choices=60)], ignore_index=True)
    t = threshold_table(m, [0.9])
    assert set(zip(t["experiment"], t["num_choices"], strict=True)) == {("exp", 5), ("other", 60)} and (t["n_total"] == 4).all()
    ref = reference_rows(matched(ROWS))
    assert ref["jev_only"]["accuracy"] == 0.5 and ref["llm_only"]["accuracy"] == 0.75 and ref["n_total"] == 4
    assert ref["jev_only"]["cost_per_1000_requests_usd_estimated"] == pytest.approx(1.0)


def canonical(rows: list[dict]) -> pd.DataFrame:
    base = {"run_id": "r", "dataset": "bitext", "locale": None, "ground_truth": "a", "candidates": ["a", "b"], "estimated_cost_usd": 0.001, "input_tokens": 100.0, "output_tokens": 10.0, "latency_ms": 100.0}
    return pd.DataFrame([{**base, "raw_row": i, **r} for i, r in enumerate(rows)])


def test_coverage_audit_counts_matches_gaps_contract_mismatches_and_unscored() -> None:
    rows = [
        {"experiment": "e", "provider": "jev", "example_id": "1", "prediction": "a", "correct": True, "confidence": 0.9},
        {"experiment": "e", "provider": "gpt-4o-mini", "example_id": "1", "prediction": "a", "correct": True, "confidence": 0.9},
        {"experiment": "e", "provider": "jev", "example_id": "2", "prediction": "a", "correct": True, "confidence": 0.9},  # no LLM row
        {"experiment": "e", "provider": "jev", "example_id": "3", "prediction": "a", "correct": True, "confidence": 0.9},
        {"experiment": "e", "provider": "gpt-4o-mini", "example_id": "3", "prediction": None, "correct": None, "confidence": None},  # LLM errored
        {"experiment": "e", "provider": "jev", "example_id": "4", "prediction": "a", "correct": True, "confidence": 0.9},
        {"experiment": "e", "provider": "gpt-4o-mini", "example_id": "4", "prediction": "a", "correct": True, "confidence": 0.9, "candidates": ["a", "b", "other"]},  # different contract
        {"experiment": "e", "provider": "gpt-4o-mini", "example_id": "5", "prediction": "a", "correct": True, "confidence": 0.9},  # no Jev row
    ]
    m, audit = build_matched(canonical(rows))
    a = audit.iloc[0]
    assert len(m) == 1 and m["example_id"].tolist() == ["1"]
    assert (a["n_jev"], a["n_llm"], a["n_matched"], a["n_jev_without_llm"], a["n_llm_without_jev"], a["n_contract_mismatch"], a["n_unscored_pair_excluded"]) == (4, 4, 1, 1, 1, 1, 1)


def test_missing_llm_estimate_reports_calls_and_cost_without_making_any(tmp_path: Path) -> None:
    (tmp_path / "noise").mkdir()
    (tmp_path / "noise" / "dataset_manifest.json").write_text(json.dumps({"n_rows": 2051}))
    can = canonical([{"experiment": "e", "provider": "gpt-4o-mini", "example_id": "1", "prediction": "a", "correct": True, "confidence": 0.9, "input_tokens": 300.0, "output_tokens": 40.0}])
    audit = pd.DataFrame([{"experiment": "e", "n_jev_without_llm": 5}])
    r = estimate_missing_llm(can, audit, tmp_path)
    assert r["canonical_examples_with_jev_but_no_llm"] == 5 and r["frozen_stress_datasets_needing_llm_predictions"]["noise"]["llm_calls_needed"] == 2051
    assert r["total_additional_llm_calls_if_all_covered"] == 2056 and r["estimated_cost_usd_all"] > 0
    assert "NOT executed" in r["action"] and "ESTIMATE" in r["estimation_basis"]


def test_run_cascade_writes_outputs_refuses_overwrite_and_states_assumptions(tmp_path: Path) -> None:
    baseline = tmp_path / "b"
    baseline.mkdir()
    rows = []
    for i in range(6):
        for prov, conf in (("jev", 0.95 if i % 2 else 0.5), ("gpt-4o-mini", 0.8)):
            rows.append({"experiment": "e", "provider": prov, "example_id": str(i), "prediction": "a", "correct": i % 3 != 0, "confidence": conf})
    canonical(rows).to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    s = run_cascade(baseline, out, [0.6, 0.9], verify=False)
    assert s["n_matched_examples"] == 6 and s["no_best_threshold_selected"] is True
    assert "sequential" in s["assumptions"]["latency"] and "Estimates, not billing" in s["assumptions"]["cost"]
    per = pd.read_csv(out / "cascade_per_example.csv")
    assert set(per["threshold"]) == {0.6, 0.9} and len(per) == 12
    assert len(pd.read_csv(out / "cascade_thresholds.csv")) == 2 and json.loads((out / "missing_llm_report.json").read_text())["action"].startswith("NOT executed")
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_cascade(baseline, out, [0.6], verify=False)
    run_cascade(baseline, tmp_path / "sw", [0.6], sweep=True, verify=False)
    sweep_table = pd.read_csv(tmp_path / "sw" / "cascade_thresholds.csv")
    assert len(sweep_table) == 50 and sweep_table["threshold"].min() == 0.5 and sweep_table["threshold"].max() == 0.99  # 0.50..0.99 in 0.01 steps
    assert set(pd.read_csv(tmp_path / "sw" / "cascade_per_example.csv")["threshold"]) == {0.6}  # per-example rows only for the requested thresholds
