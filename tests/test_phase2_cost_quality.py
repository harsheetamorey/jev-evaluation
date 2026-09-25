"""Tests for the cost x quality comparison. No model calls, no network."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.baseline import BaselineError
from phase2.cost_quality import (
    LATENCY_MEASURED,
    LATENCY_SIMULATED,
    assert_same_population,
    cascade_point,
    mark_frontier,
    oracle_point,
    population_fingerprint,
    run_cost_quality,
    single_provider_point,
    tradeoffs,
    workload_points,
)

WL = {"workload": "exp|k=5", "experiment": "exp", "num_choices": 5, "dataset": "bitext"}


def matched(rows: list[tuple] | None = None) -> pd.DataFrame:
    """(jev_pred, jev_conf, jev_ok, jev_lat, jev_cost, llm_pred, llm_ok, llm_lat, llm_cost)"""
    rows = rows or [
        ("a", 0.95, True, 100.0, 0.001, "a", True, 900.0, 0.010),
        ("b", 0.90, False, 110.0, 0.001, "a", True, 800.0, 0.010),
        ("b", 0.60, False, 120.0, 0.001, "a", True, 700.0, 0.010),
        ("a", 0.50, True, 130.0, 0.001, "b", False, 600.0, 0.010),
    ]
    return pd.DataFrame([{"experiment": "exp", "example_id": f"e{i}", "locale": None, "num_choices": 5, "dataset": "bitext", "ground_truth": "a", "text": "hi", "jev_prediction": jp, "jev_confidence": jc, "jev_correct": jok, "jev_latency_ms": jl, "jev_estimated_cost_usd": jcost,
                         "llm_prediction": lp, "llm_confidence": 0.8, "llm_correct": lok, "llm_latency_ms": ll, "llm_estimated_cost_usd": lcost} for i, (jp, jc, jok, jl, jcost, lp, lok, ll, lcost) in enumerate(rows)])


def test_population_fingerprint_is_order_independent_and_detects_a_different_population() -> None:
    m = matched()
    assert population_fingerprint(m) == population_fingerprint(m.iloc[::-1])
    assert population_fingerprint(m) != population_fingerprint(m.iloc[:3])


def test_same_population_is_enforced_across_architecture_points() -> None:
    pts = workload_points(matched(), WL)
    assert len({p["population_sha256"] for p in pts}) == 1 and {p["n"] for p in pts} == {4}
    assert_same_population(pts)
    bad = [dict(pts[0]), {**pts[1], "population_sha256": "different", "n": 3}]
    with pytest.raises(ValueError, match="different populations"):
        assert_same_population(bad)


def test_single_provider_cost_per_1000_cost_per_correct_and_measured_latency() -> None:
    p = single_provider_point(matched(), WL, "sha", "jev", "jev_only")
    assert p["accuracy"] == 0.5 and p["n_correct"] == 2 and p["cost_complete"] is True
    assert p["cost_per_1000_requests_usd"] == pytest.approx(1.0)  # 4 requests x $0.001 -> $0.001/request = $1 per 1000
    assert p["cost_per_correct_decision_usd"] == pytest.approx(0.004 / 2)
    assert p["latency_provenance"] == LATENCY_MEASURED and p["latency_n"] == 4
    assert p["latency_p50_ms"] == pytest.approx(pd.Series([100, 110, 120, 130]).quantile(0.5))
    llm = single_provider_point(matched(), WL, "sha", "llm", "llm_only")
    assert llm["accuracy"] == 0.75 and llm["cost_per_1000_requests_usd"] == pytest.approx(10.0)


def test_cascade_point_costs_jev_everywhere_llm_only_on_fallbacks_with_simulated_latency() -> None:
    p = cascade_point(matched(), WL, "sha", 0.90)
    assert p["accuracy"] == 0.5 and p["automation_coverage"] == 1.0
    assert p["cost_per_1000_requests_usd"] == pytest.approx((4 * 0.001 + 2 * 0.010) / 4 * 1000)
    assert p["latency_provenance"] == LATENCY_SIMULATED and p["latency_p95_ms"] == pytest.approx(pd.Series([100, 110, 820, 730]).quantile(0.95))
    assert p["cost_per_correct_decision_usd"] == pytest.approx((4 * 0.001 + 2 * 0.010) / 2)


def test_missing_cost_or_latency_is_unavailable_never_zero() -> None:
    m = matched()
    m.loc[0, "jev_estimated_cost_usd"] = None
    p = single_provider_point(m, WL, "sha", "jev", "jev_only")
    assert p["cost_per_1000_requests_usd"] is None and p["cost_complete"] is False and p["cost_per_correct_decision_usd"] is None
    assert cascade_point(m, WL, "sha", 0.90)["cost_per_1000_requests_usd"] is None
    m2 = matched()
    m2.loc[1, "jev_latency_ms"] = None
    q = single_provider_point(m2, WL, "sha", "jev", "jev_only")
    assert q["latency_n"] == 3  # latency aggregates only over the measurements that exist, and says how many


def test_oracle_architecture_is_labelled_partial_and_never_claims_total_cost() -> None:
    p = oracle_point(matched(), WL, "sha", 0.90, 0.9)  # the fake LLM confidence is 0.8, so its fallbacks are not validated and reach the oracle
    assert p["includes_oracle"] is True and p["cost_complete"] is False and p["cost_per_correct_decision_usd"] is None
    assert p["oracle_fallback_rate"] > 0 and p["automation_coverage"] < 1.0 and "NOT total system cost" in p["note"]
    assert p["accuracy"] > single_provider_point(matched(), WL, "sha", "jev", "jev_only")["accuracy"]  # inflated by construction: why it is excluded from the frontier
    assert "oracle latency NA" in p["latency_provenance"]


def test_workload_points_cover_every_architecture_and_flag_unavailable_rules_and_degenerate_duplicates() -> None:
    pts = pd.DataFrame(workload_points(matched(), WL))
    assert {"rules_only", "jev_only", "llm_only", "jev_to_llm", "rules_jev_llm", "rules_jev_llm_oracle"} == set(pts["architecture"])
    rules = pts[pts["architecture"] == "rules_only"].iloc[0]
    assert rules["status"] == "not_available" and "rules_client.py is empty" in rules["note"] and pd.isna(rules["accuracy"])
    dup = pts[pts["architecture"] == "rules_jev_llm"]
    assert dup["degenerate_reason"].notna().all() and set(dup["accuracy"]) == set(pts[pts["architecture"] == "jev_to_llm"]["accuracy"])
    assert "estimated" in pts["cost_provenance"].iloc[0].lower() and "not billing" in pts["cost_provenance"].iloc[0].lower()


def test_frontier_flag_is_descriptive_and_excludes_oracle_degenerate_and_incomplete_points() -> None:
    df = mark_frontier(pd.DataFrame(workload_points(matched(), WL)))
    flagged = df[df["on_automated_cost_accuracy_frontier"]]
    assert len(flagged) >= 1 and not flagged["includes_oracle"].any() and flagged["degenerate_reason"].isna().all() and (flagged["status"] == "computed").all()
    by = df.set_index(["architecture", "config"])["on_automated_cost_accuracy_frontier"]
    assert by[("jev_only", "every request")]  # cheapest point: nothing is both cheaper-or-equal and as accurate
    assert not by[("llm_only", "every request")]  # jev_to_llm at 0.95 matches its 0.75 accuracy at lower cost, so it dominates llm_only here
    assert by[("jev_to_llm", "jev_threshold=0.95")]
    assert not by[("jev_to_llm", "jev_threshold=0.90")]  # dominated by jev_only: same accuracy, higher cost


def test_tradeoffs_state_differences_without_declaring_a_winner() -> None:
    t = tradeoffs(pd.DataFrame(workload_points(matched(), WL)))[0]
    assert t["accuracy_difference_jev_minus_llm"] == pytest.approx(0.5 - 0.75) and t["cost_per_1000_jev_only_usd"] < t["cost_per_1000_llm_only_usd"]
    assert "no architecture is declared best" in t["note"] and not any("best" in k or "winner" in k for k in t)


def test_run_cost_quality_writes_outputs_per_workload_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline = tmp_path / "b"
    baseline.mkdir()
    rows = []
    for exp, k_cands in (("e1", ["a", "b"]), ("e2", ["a", "b", "c"])):  # two workloads that must not be mixed
        for i in range(4):
            for prov, conf in (("jev", 0.95 if i % 2 else 0.5), ("gpt-4o-mini", 0.8)):
                rows.append({"raw_row": len(rows), "run_id": exp, "experiment": exp, "provider": prov, "example_id": str(i), "dataset": "bitext", "locale": None, "ground_truth": "a", "candidates": k_cands, "prediction": "a", "correct": i != 0,
                             "confidence": conf, "latency_ms": 100.0, "estimated_cost_usd": 0.001, "text": "hi"})
    pd.DataFrame(rows).to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    s = run_cost_quality(baseline, out, verify=False)
    assert s["workloads"] == ["e1|k=2", "e2|k=3"] and s["no_best_architecture_selected"] is True and s["rules"]["available"] is False
    frontier = pd.read_csv(out / "architecture_frontier.csv")
    assert {"population_sha256", "cost_provenance", "latency_provenance", "n"} <= set(frontier.columns)
    for _, g in frontier.groupby("workload"):
        assert g["population_sha256"].nunique() == 1  # one population per workload
    assert frontier.groupby("workload")["population_sha256"].first().nunique() == 2  # ...and different workloads are kept apart
    assert {"cost_per_1000_requests_usd", "accuracy", "latency_p95_ms"} <= set(pd.read_csv(out / "cost_accuracy_points.csv").columns)
    assert "not a recommendation" in json.loads((out / "frontier_summary.json").read_text())["frontier_flag"]
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_cost_quality(baseline, out, verify=False)
