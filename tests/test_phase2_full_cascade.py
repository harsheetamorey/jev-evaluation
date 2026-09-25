"""Tests for every routing path of the offline full cascade. No model calls, no network."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.baseline import BaselineError
from phase2.full_cascade import (
    HANDLERS,
    CascadeConfig,
    NoRules,
    grid,
    route_request,
    run_full_cascade,
    simulate_full,
    summarize_full,
)


def req(**over) -> pd.Series:
    base = {"experiment": "exp", "example_id": "e1", "locale": None, "num_choices": 5, "ground_truth": "a", "text": "please cancel",
            "jev_prediction": "a", "jev_confidence": 0.95, "jev_correct": True, "jev_latency_ms": 100.0, "jev_estimated_cost_usd": 0.001,
            "llm_prediction": "a", "llm_confidence": 0.8, "llm_correct": True, "llm_latency_ms": 900.0, "llm_estimated_cost_usd": 0.010}
    return pd.Series({**base, **over})


class KeywordRules:
    """A stand-in rules handler used only in tests: handles messages containing 'cancel'."""

    def handle(self, request: pd.Series) -> str | None:
        return "a" if "cancel" in str(request.get("text", "")) else None


CFG = CascadeConfig(jev_threshold=0.90, llm_threshold=0.5)


def test_real_repository_rules_baseline_handles_nothing() -> None:
    assert NoRules().handle(req()) is None
    r = route_request(req(), CFG, NoRules())
    assert r["final_handler"] == "jev" and r["stages_visited"] == "rules|jev"


def test_rules_path_short_circuits_and_uses_no_model_stage() -> None:
    r = route_request(req(), CFG, KeywordRules())
    assert (r["final_handler"], r["stage_reached"], r["final_prediction"], r["final_correct"]) == ("rules", "rules", "a", True)
    assert r["stages_visited"] == "rules" and pd.isna(r["jev_latency_ms"]) and pd.isna(r["jev_cost_usd"]) and pd.isna(r["path_latency_ms"]) is True
    wrong = route_request(req(ground_truth="b"), CFG, KeywordRules())  # a rule that answers wrongly is a rule error
    assert wrong["final_handler"] == "rules" and wrong["final_correct"] is False


def test_jev_path_accepts_at_or_above_threshold_and_records_stage_latency_and_cost() -> None:
    r = route_request(req(jev_confidence=0.90), CFG, NoRules())
    assert (r["final_handler"], r["stage_reached"], r["fallback_reason"]) == ("jev", "jev", "rules_cannot_handle")
    assert r["path_latency_ms"] == 100.0 and r["automated_stage_cost_usd_estimated"] == pytest.approx(0.001) and pd.isna(r["llm_latency_ms"])
    assert r["jev_confidence"] == 0.90


def test_llm_path_when_jev_is_unsure_and_the_llm_is_validated() -> None:
    r = route_request(req(jev_confidence=0.60, jev_prediction="b", jev_correct=False), CFG, NoRules())
    assert (r["final_handler"], r["stage_reached"], r["final_prediction"], r["final_correct"]) == ("llm", "llm", "a", True)
    assert r["fallback_reason"] == "rules_cannot_handle;jev_below_threshold"
    assert r["path_latency_ms"] == 1000.0 and r["automated_stage_cost_usd_estimated"] == pytest.approx(0.011)  # sequential: Jev then LLM


def test_oracle_path_when_llm_is_not_validated_returns_ground_truth_and_has_no_latency_or_cost() -> None:
    r = route_request(req(jev_confidence=0.6, llm_confidence=0.3, llm_prediction="b", llm_correct=False), CFG, NoRules())
    assert (r["final_handler"], r["stage_reached"], r["final_prediction"], r["final_correct"]) == ("oracle", "oracle", "a", True)
    assert "llm_not_validated_confidence" in r["fallback_reason"] and pd.isna(r["oracle_latency_ms"]) and pd.isna(r["oracle_cost_usd"])
    assert pd.isna(r["path_latency_ms"])  # no defensible total path latency once the oracle is involved
    assert r["automated_stage_latency_ms"] == 1000.0  # the automated part is still recorded, separately


def test_oracle_reached_when_llm_prediction_is_missing_or_llm_disabled() -> None:
    missing = route_request(req(jev_confidence=0.6, llm_prediction=None, llm_correct=None), CFG, NoRules())
    assert missing["final_handler"] == "oracle" and "llm_missing_prediction" in missing["fallback_reason"]
    no_llm = route_request(req(jev_confidence=0.6), CascadeConfig(llm_enabled=False), NoRules())
    assert no_llm["final_handler"] == "oracle" and "llm_disabled" in no_llm["fallback_reason"] and no_llm["stages_visited"] == "rules|jev|oracle"
    jev_missing = route_request(req(jev_prediction=None, jev_confidence=None, jev_correct=None), CFG, NoRules())
    assert jev_missing["final_handler"] == "llm" and "jev_missing_prediction" in jev_missing["fallback_reason"]


def test_disabling_the_oracle_leaves_unresolved_requests_unhandled_not_correct() -> None:
    r = route_request(req(jev_confidence=0.6, llm_confidence=0.3), CascadeConfig(oracle_enabled=False), NoRules())
    assert r["final_handler"] == "unhandled" and r["final_prediction"] is None and r["final_correct"] is None
    s = summarize_full(simulate_full(pd.DataFrame([req(jev_confidence=0.6, llm_confidence=0.3), req()]), CascadeConfig(oracle_enabled=False)))
    assert (s["n_unhandled"], s["n_jev"], s["overall_accuracy_including_oracle"]) == (1, 1, 1.0)  # accuracy over handled requests only; n shown


def test_llm_threshold_none_accepts_any_valid_llm_output_but_missing_confidence_cannot_be_validated() -> None:
    cfg_any = CascadeConfig(llm_threshold=None)
    assert route_request(req(jev_confidence=0.6, llm_confidence=0.01), cfg_any, NoRules())["final_handler"] == "llm"
    assert route_request(req(jev_confidence=0.6, llm_confidence=None), CFG, NoRules())["final_handler"] == "oracle"


def routes() -> pd.DataFrame:
    return pd.DataFrame([
        req(text="please cancel"),                                                                         # rules
        req(text="hi", example_id="e2"),                                                                    # jev, correct
        req(text="hi", example_id="e3", jev_confidence=0.95, jev_correct=False, jev_prediction="b"),        # jev, wrong (confident error)
        req(text="hi", example_id="e4", jev_confidence=0.5, llm_correct=True),                               # llm, correct
        req(text="hi", example_id="e5", jev_confidence=0.5, llm_correct=False, llm_prediction="b"),          # llm, wrong
        req(text="hi", example_id="e6", jev_confidence=0.5, llm_confidence=0.1),                             # oracle
    ])


def test_coverage_accounting_and_stage_errors_cover_every_path() -> None:
    s = summarize_full(simulate_full(routes(), CFG, KeywordRules()))
    assert (s["n_total"], s["n_rules"], s["n_jev"], s["n_llm"], s["n_oracle"], s["n_unhandled"]) == (6, 1, 2, 2, 1, 0)
    assert sum(s[f"n_{h}"] for h in HANDLERS) + s["n_unhandled"] == s["n_total"]
    assert sum(s[f"coverage_{h}"] for h in HANDLERS) == pytest.approx(1.0)
    assert s["automation_coverage_excluding_oracle"] == pytest.approx(5 / 6)
    assert s["stage_errors"] == {"rules": 0, "jev": 1, "llm": 1}
    assert s["automated_accuracy_excluding_oracle"] == pytest.approx(3 / 5)  # rules ok, jev ok, jev wrong, llm ok, llm wrong
    assert s["overall_accuracy_including_oracle"] == pytest.approx(4 / 6) and "NOT a measured human" in s["overall_accuracy_note"]
    assert s["path_latency_n"] == 4  # only the 2 Jev + 2 LLM requests: rules latency is not modeled and the oracle request has no total path latency
    assert s["fallback_reasons"]["jev_below_threshold"] == 3


def test_real_rules_give_zero_rule_coverage_and_nothing_is_invented() -> None:
    s = summarize_full(simulate_full(routes(), CFG, NoRules()))
    assert s["n_rules"] == 0 and s["coverage_rules"] == 0.0 and s["fallback_reasons"]["rules_cannot_handle"] == 6


def test_grid_is_configurable_per_experiment_and_does_not_pick_a_winner() -> None:
    g = grid(routes().assign(experiment="exp"), [0.6, 0.9], [None, 0.5], NoRules())
    assert len(g) == 4 and set(g["jev_threshold"]) == {0.6, 0.9} and set(g["experiment"]) == {"exp"}
    assert not any("best" in c for c in g.columns)
    low, high = g[(g["jev_threshold"] == 0.6) & (g["llm_threshold"].isna())].iloc[0], g[(g["jev_threshold"] == 0.9) & (g["llm_threshold"].isna())].iloc[0]
    assert low["n_jev"] >= high["n_jev"]


def test_run_full_cascade_writes_outputs_states_rules_reality_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline = tmp_path / "b"
    baseline.mkdir()
    rows = []
    for i in range(6):
        for prov, conf in (("jev", 0.95 if i % 2 else 0.5), ("gpt-4o-mini", 0.8)):
            rows.append({"raw_row": len(rows), "run_id": "r", "experiment": "e", "provider": prov, "example_id": str(i), "dataset": "bitext", "locale": None, "ground_truth": "a", "candidates": ["a", "b"], "prediction": "a", "correct": i % 3 != 0,
                         "confidence": conf, "latency_ms": 100.0, "estimated_cost_usd": 0.001, "text": f"msg {i}"})
    pd.DataFrame(rows).to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    s = run_full_cascade(baseline, out, jev_thresholds=[0.6, 0.9], llm_thresholds=[0.5], verify=False)
    assert s["n_requests"] == 6 and s["rules"]["actual_coverage"] == 0.0 and "empty" in s["rules"]["reason"]
    assert "NOT a measured human" in s["assumptions"]["oracle"] and s["thresholds_are_configuration_not_recommendations"] is True
    per = pd.read_csv(out / "full_cascade_per_request.csv")
    assert {"stage_reached", "final_handler", "final_prediction", "final_correct", "jev_confidence", "fallback_reason", "path_latency_ms"} <= set(per.columns) and len(per) == 6
    assert len(pd.read_csv(out / "full_cascade_grid.csv")) == 2 and json.loads((out / "full_cascade_summary.json").read_text())["per_request_config"]["oracle_enabled"] is True
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_full_cascade(baseline, out, verify=False)
