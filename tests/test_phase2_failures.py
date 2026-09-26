"""Tests for the failure taxonomy / Failure Museum data model. No model calls, no network."""

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset_loaders.massive import LOCALES
from phase2.baseline import BaselineError
from phase2.failures import (
    ADAPTERS,
    json_safe,
    COLUMNS,
    FAILURE_TAGS,
    assemble,
    baseline_failures,
    confidence_tags,
    failure_record,
    make_failure_id,
    run_failures,
    stress_failures,
    summarize,
    validate_failures,
)


def rec(**over):
    base = dict(experiment="e", dataset="d", provider="jev", source_example_id="s1", source_file="f.parquet", source_key="k1", input_text="hi", transformed_input=None, ground_truth="a", prediction="b", confidence=0.5, cause_tags=[], primary=None, condition=None, locale=None, metadata={})
    return failure_record(**{**base, **over})


def test_failure_ids_are_deterministic_and_change_with_any_provenance_part() -> None:
    a = make_failure_id("e", "jev", "f", "k")
    assert a == make_failure_id("e", "jev", "f", "k") and a.startswith("F-") and len(a) == 18
    assert len({a, make_failure_id("e2", "jev", "f", "k"), make_failure_id("e", "gpt", "f", "k"), make_failure_id("e", "jev", "f2", "k"), make_failure_id("e", "jev", "f", "k2")}) == 5


def test_confidence_tags_use_configurable_thresholds_inclusively() -> None:
    assert confidence_tags(0.95) == ["high_confidence_wrong"] and confidence_tags(0.90) == ["high_confidence_wrong"]
    assert confidence_tags(0.60) == ["low_confidence_wrong"] and confidence_tags(0.75) == []
    assert confidence_tags(0.75, high=0.7) == ["high_confidence_wrong"] and confidence_tags(0.75, low=0.8) == ["low_confidence_wrong"]
    assert confidence_tags(None) == [] and confidence_tags(float("nan")) == []


def test_a_failure_can_carry_several_tags_and_provenance_is_kept() -> None:
    r = rec(cause_tags=["adversarial_manipulation"], confidence=0.97, primary="adversarial_manipulation", condition="role_play", metadata={"target_label": "x"}, run_id="run1")
    assert r["failure_tags"] == ["adversarial_manipulation", "high_confidence_wrong"]  # cause + confidence, no forced single explanation
    assert (r["source_file"], r["source_key"], r["run_id"], r["stress_condition"], r["primary_failure_type"]) == ("f.parquet", "k1", "run1", "role_play", "adversarial_manipulation")
    assert rec(cause_tags=["high_confidence_wrong"], confidence=0.95)["failure_tags"] == ["high_confidence_wrong"]  # no duplicate tags
    assert rec(confidence=None)["confidence"] is None and rec()["primary_failure_type"] is None


def test_required_fields_and_unknown_tags_are_enforced() -> None:
    df = assemble([rec(), rec(source_key="k2")])
    assert list(df.columns) == COLUMNS
    validate_failures(df)
    with pytest.raises(ValueError, match="required field 'prediction'"):
        validate_failures(df.assign(prediction=[None, "x"]))
    with pytest.raises(ValueError, match="unknown failure tags"):
        validate_failures(df.assign(failure_tags=[["made_up"], []]))
    with pytest.raises(ValueError, match="missing columns"):
        validate_failures(df.drop(columns=["source_file"]))


def test_deduplication_keeps_one_row_per_failure_id() -> None:
    first, again = rec(prediction="b"), rec(prediction="c")  # same provenance -> same id
    assert first["failure_id"] == again["failure_id"]
    df = assemble([first, again, rec(source_key="k2")])
    assert len(df) == 2 and df[df["source_key"] == "k1"]["prediction"].item() == "c"  # the latest record wins


def canonical_rows() -> pd.DataFrame:
    rows, raw = [], 0

    def add(exp, provider, ex, loc, gt, pred, conf, run="r1", text="txt"):
        nonlocal raw
        ok = None if pred is None else pred == gt
        rows.append({"raw_row": raw, "run_id": run, "experiment": exp, "provider": provider, "example_id": ex, "dataset": "massive" if loc else "bitext", "locale": loc, "ground_truth": gt, "prediction": pred, "correct": ok, "confidence": conf, "candidates": ["a", "b"], "text": text})
        raw += 1

    add("bitext_hard_choice", "jev", "b1", None, "a", "b", 0.97)  # high-confidence wrong
    add("bitext_hard_choice", "jev", "b2", None, "a", "b", 0.30)  # low-confidence wrong
    add("bitext_hard_choice", "jev", "b3", None, "a", "b", 0.75)  # wrong, no confidence tag -> unclassified
    add("bitext_hard_choice", "jev", "b4", None, "a", "a", 0.9)  # correct: not a failure
    add("bitext_hard_choice", "jev", "b5", None, "a", None, None)  # errored call: not a classification failure
    for loc in LOCALES:  # one example wrong ONLY in ja-JP -> language_specific
        add("multilingual", "jev", "m1", loc, "x", "y" if loc == "ja-JP" else "x", 0.8)
    for loc in LOCALES:  # one example wrong in every language -> not attributed to language
        add("multilingual", "jev", "m2", loc, "x", "y", 0.8)
    add("bitext_hard_choice", "jev", "b1", None, "a", "a", 0.9, run="r2")  # later re-measurement of b1 (now correct)
    return pd.DataFrame(rows)


def test_baseline_failures_tag_by_confidence_and_language_pattern_only() -> None:
    fails = baseline_failures(canonical_rows(), "canonical.parquet")
    by_key = {(f["experiment"], f["source_key"]): f for f in fails}
    assert ("bitext_hard_choice", "b1") not in by_key  # the latest measurement was correct, so it is not a failure
    assert by_key[("bitext_hard_choice", "b2")]["failure_tags"] == ["low_confidence_wrong"]
    assert by_key[("bitext_hard_choice", "b3")]["failure_tags"] == []  # cannot be classified
    assert ("bitext_hard_choice", "b4") not in by_key and ("bitext_hard_choice", "b5") not in by_key
    assert by_key[("multilingual", "m1:ja-JP")]["failure_tags"] == ["language_specific"]
    assert by_key[("multilingual", "m2:en-US")]["failure_tags"] == []  # wrong in all 8 languages -> no language attribution
    assert all(f["primary_failure_type"] is None and f["source_file"] == "canonical.parquet" for f in fails)
    assert len({f["failure_id"] for f in fails}) == len(fails)


def test_high_confidence_baseline_failure_is_tagged() -> None:
    df = canonical_rows().query("run_id == 'r1'")
    f = {x["source_key"]: x for x in baseline_failures(df, "c.parquet")}
    assert f["b1"]["failure_tags"] == ["high_confidence_wrong"] and f["b1"]["input"] == "txt"


def stress(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(dataset="stress:x", locale=None, run_id="r")


def test_adversarial_failures_get_the_condition_established_cause() -> None:
    df = stress([
        {"provider": "jev", "source_example_id": "s1", "variant_id": "s1:clean", "attack_type": "none", "text": "clean", "clean_input": "clean", "ground_truth": "a", "prediction": "a", "correct": True, "confidence": 0.9, "target_label": "t", "intent": "a"},
        {"provider": "jev", "source_example_id": "s1", "variant_id": "s1:role_play", "attack_type": "role_play", "text": "clean\n\nattack", "clean_input": "clean", "ground_truth": "a", "prediction": "t", "correct": False, "confidence": 0.95, "target_label": "t", "intent": "a"},
    ])
    (f,) = stress_failures("adversarial", df, "adv.parquet")
    assert f["primary_failure_type"] == "adversarial_manipulation" and f["failure_tags"] == ["adversarial_manipulation", "high_confidence_wrong"]
    assert f["input"] == "clean" and f["transformed_input"] == "clean\n\nattack" and f["stress_condition"] == "role_play" and f["source_key"] == "s1:role_play"
    assert f["metadata"]["target_label"] == "t"


def test_overlap_tag_only_when_the_condition_says_overlapping() -> None:
    rows = [{"provider": "jev", "source_example_id": "s1", "variant_id": f"s1:{t}", "choice_set_type": t, "number_of_choices": 3, "choice_set_id": f"s1:{t}", "family": "refund", "text": "x", "ground_truth": "a", "prediction": "b", "correct": False, "confidence": 0.7} for t in ("distinct", "overlapping")]
    tags = {f["stress_condition"]: f["failure_tags"] for f in stress_failures("choice_overlap", stress(rows), "co.parquet")}
    assert tags == {"distinct": [], "overlapping": ["semantic_overlap"]}  # never inferred from label similarity


def test_context_conditions_ambiguity_ood_and_noise_rules() -> None:
    ctx = stress([{"provider": "jev", "source_example_id": "t1", "variant_id": f"t1:{c}", "context_condition": c, "base_message": "vague", "text": "vague", "ground_truth": "a", "prediction": "b", "correct": False, "confidence": 0.7, "relevance_cues": ["x"], "intent": "a"} for c in ("no_context", "irrelevant_context", "relevant_context")])
    got = {f["stress_condition"]: f["primary_failure_type"] for f in stress_failures("context_relevance", ctx, "c.parquet")}
    assert got == {"no_context": "insufficient_context", "irrelevant_context": "irrelevant_context", "relevant_context": None}

    amb = stress([{"provider": "jev", "source_example_id": "s1", "variant_id": f"s1:{lvl}", "ambiguity_type": lvl, "original_text": "orig", "text": "orig", "ground_truth": gt, "prediction": "z", "correct": None if gt is None else False, "confidence": 0.7, "acceptable_intents": json.dumps(acc), "primary_expected_intent": gt, "family": "refund"}
                  for lvl, gt, acc in (("clear", "a", ["a"]), ("underspecified", None, ["a", "b"]))])
    got = {f["stress_condition"]: f["primary_failure_type"] for f in stress_failures("ambiguity", amb, "a.parquet")}
    assert got == {"clear": None, "underspecified": "ambiguous_intent"}  # a non-clear miss is tagged; a clear miss is not

    ood = stress([{"provider": "jev", "source_example_id": p, "variant_id": f"{p}:{c}", "population": p, "contract_type": c, "text": "play jazz", "ground_truth": None, "prediction": pred, "correct": None, "confidence": 0.9, "massive_intent": "m", "matched_control_id": "c"} for p, c, pred in (("ood", "forced_choice", "get_refund"), ("ood", "explicit_fallback", "other"), ("ood", "explicit_fallback", "get_refund"))])
    keys = {f["source_key"] for f in stress_failures("ood", ood.assign(variant_id=["v1", "v2", "v3"]), "o.parquet")}
    assert keys == {"v1", "v3"}  # choosing 'other' for an OOD input is not a failure

    noise = stress([{"provider": "jev", "source_example_id": "s1", "variant_id": f"s1:{i}", "noise_type": "typos", "severity": sev, "semantic_integrity": integ, "original_text": "orig", "text": "txt", "ground_truth": "a", "prediction": "b", "correct": False, "confidence": 0.7, "intent": "a"} for i, (sev, integ) in enumerate([(0, "valid"), (3, "valid"), (3, "questionable")])])
    got = [(f["source_key"], f["primary_failure_type"]) for f in stress_failures("noise", noise, "n.parquet")]
    assert got == [("s1:1", "surface_noise"), ("s1:2", None)]  # clean row excluded; meaning-corrupted noise is not blamed for the failure


def test_every_adapter_only_emits_known_tags() -> None:
    assert set(ADAPTERS) == {"ambiguity", "stability", "noise", "context_pollution", "context_relevance", "ood", "adversarial", "choice_overlap"}
    assert set(FAILURE_TAGS) >= {"ambiguous_intent", "semantic_overlap", "insufficient_context", "irrelevant_context", "ood_input", "language_specific", "surface_noise", "adversarial_manipulation", "high_confidence_wrong", "low_confidence_wrong"}


def test_summary_reports_counts_unclassified_and_missing_experiments() -> None:
    df = assemble([rec(source_key="a", confidence=0.95), rec(source_key="b", confidence=0.3, cause_tags=["ood_input"]), rec(source_key="c", confidence=0.75)])
    s = summarize(df, ["noise"], 0.9, 0.6)
    assert s["total_failures"] == 3 and s["by_failure_tag"]["high_confidence_wrong"] == 1 and s["by_failure_tag"]["ood_input"] == 1
    assert s["n_unclassified_no_tag"] == 1 and s["n_with_multiple_tags"] == 1 and s["experiments_with_no_recorded_results"] == ["noise"]
    assert s["high_confidence_wrong_examples"][0]["confidence"] == 0.95 and "not scientifically optimal" in s["thresholds"]["note"]


def test_run_failures_end_to_end_writes_outputs_reports_missing_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline, phase2 = tmp_path / "b", tmp_path / "p"
    baseline.mkdir()
    canonical_rows().to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    s = run_failures(baseline, phase2, out, verify=False)
    assert s["total_failures"] > 0 and set(s["experiments_with_no_recorded_results"]) == set(ADAPTERS)  # no stress results yet: reported, not invented
    lines = [json.loads(line) for line in (out / "failures.jsonl").read_text().splitlines()]
    assert len(lines) == s["total_failures"] and all(isinstance(x["failure_tags"], list) for x in lines)
    assert (out / "failures.csv").exists() and (out / "failures_summary.json").exists()
    again = tmp_path / "out2"
    run_failures(baseline, phase2, again, verify=False)
    rerun_ids = [json.loads(line)["failure_id"] for line in (again / "failures.jsonl").read_text().splitlines()]
    assert rerun_ids == [x["failure_id"] for x in lines]  # same inputs -> same ids, same order
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_failures(baseline, phase2, out, verify=False)


def test_jsonl_output_is_strict_json_with_no_nan_tokens(tmp_path: Path) -> None:
    baseline = tmp_path / "b"
    baseline.mkdir()
    canonical_rows().to_parquet(baseline / "canonical_results.parquet", index=False)  # non-multilingual rows have a NaN/None locale
    out = tmp_path / "out"
    run_failures(baseline, tmp_path / "p", out, verify=False)

    def reject(token: str):
        raise ValueError(f"non-strict JSON constant {token}")

    for line in (out / "failures.jsonl").read_text().splitlines():
        rec = json.loads(line, parse_constant=reject)  # NaN / Infinity would raise here
        assert rec["locale"] is None or isinstance(rec["locale"], str)


def test_json_safe_converts_nan_and_numpy_values() -> None:
    import numpy as np

    got = json_safe({"a": float("nan"), "b": np.float64(0.5), "c": [np.int64(3), float("nan")], "d": None, "e": "x"})
    assert got == {"a": None, "b": 0.5, "c": [3, None], "d": None, "e": "x"}
    json.dumps(got, allow_nan=False)
