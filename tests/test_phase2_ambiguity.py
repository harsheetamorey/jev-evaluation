"""Tests for the ambiguity stress dataset and metrics. No network or API access."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.ambiguity import (
    FAMILIES,
    LEGACY_LEVELS,
    LEVEL_TO_TYPE,
    TYPES,
    SIDE_QUESTION,
    acceptable_set_correct,
    analyze,
    build,
    build_variants,
    describe,
    enrich,
    pick_partner,
    shared_families,
    validate_type,
)
from phase2.stress import StressError, load_frozen_dataset, validate_rows, write_frozen_dataset


def test_ambiguity_type_validation() -> None:
    for kind in TYPES:
        assert validate_type(kind) == kind
    with pytest.raises(StressError, match="Unknown ambiguity_type"):
        validate_type("MILDLY_AMBIGUOUS")
    assert TYPES == ("clear", "competing_intents", "underspecified")
    assert LEVEL_TO_TYPE == {"CLEAR": "clear", "SLIGHTLY_AMBIGUOUS": "competing_intents", "HIGHLY_AMBIGUOUS": "underspecified"}


def test_every_intent_has_a_side_question_and_families_reference_known_intents() -> None:
    members = {m for ms, _ in FAMILIES.values() for m in ms}
    assert members <= set(SIDE_QUESTION) and len(SIDE_QUESTION) == 27


def test_partner_is_a_real_distractor_sharing_a_family() -> None:
    choices = ["get_refund", "track_refund", "check_cancellation_fee", "place_order", "complaint"]
    partner, family = pick_partner("src-1", "get_refund", choices)
    assert partner in choices and partner != "get_refund" and family in shared_families("get_refund", partner)
    assert pick_partner("src-1", "complaint", ["complaint", "review", "place_order"]) is None  # no overlapping family: no fake ambiguity
    assert pick_partner("src-1", "get_refund", choices) == (partner, family)  # deterministic


def test_variants_keep_source_id_labels_and_do_not_force_a_single_truth() -> None:
    v = {r["ambiguity_type"]: r for r in build_variants("src-1", "i want my money back", ["get_refund", "track_refund", "place_order", "complaint", "review"], "get_refund", "track_refund", "refund")}
    assert {r["source_example_id"] for r in v.values()} == {"src-1"} and set(v) == set(TYPES)
    assert v["clear"]["text"] == "i want my money back" and v["clear"]["acceptable_intents"] == ["get_refund"]
    assert v["competing_intents"]["text"].startswith("i want my money back") and v["competing_intents"]["primary_expected_intent"] == "get_refund"
    assert v["competing_intents"]["acceptable_intents"] == ["get_refund", "track_refund"]
    high = v["underspecified"]
    assert high["ground_truth"] is None and high["primary_expected_intent"] is None
    assert high["acceptable_intents"] == ["get_refund", "track_refund"]  # family members present in the choice set (not check_refund_policy)
    assert all(r["original_text"] == "i want my money back" and r["original_ground_truth"] == "get_refund" for r in v.values())
    assert {r["ambiguity_level"] for r in v.values()} == set(LEGACY_LEVELS)  # legacy alias kept for older readers
    assert {r["variant_id"] for r in v.values()} == {f"src-1:{t}" for t in TYPES}
    validate_rows(list(v.values()))


def test_acceptable_set_handling() -> None:
    assert acceptable_set_correct("a", ["a", "b"]) is True and acceptable_set_correct("c", ["a", "b"]) is False
    assert acceptable_set_correct(None, ["a"]) is None


def test_built_dataset_is_deterministic_complete_and_loads_from_a_frozen_dir(tmp_path: Path) -> None:
    rows, meta = build()
    rows2, _ = build()
    assert rows == rows2  # same bytes every time
    assert len(rows) == 300 and len({r["source_example_id"] for r in rows}) == 100
    counts = pd.Series([r["ambiguity_type"] for r in rows]).value_counts()
    assert set(counts) == {100} and set(counts.index) == set(TYPES)
    assert all(LEVEL_TO_TYPE[r["ambiguity_level"]] == r["ambiguity_type"] for r in rows) and meta["ordinal_scale_claimed"] is False
    for r in rows:  # ground truth is always among the choices; ambiguous items never invent a single truth
        assert set(r["acceptable_intents"]) <= set(r["candidates"]) and r["original_ground_truth"] in r["candidates"]
        assert (r["ground_truth"] is None) == (r["ambiguity_type"] == "underspecified")
    write_frozen_dataset(tmp_path, rows, meta)
    loaded, manifest = load_frozen_dataset(tmp_path)
    assert loaded == rows and manifest["n_rows"] == 300 and manifest["review_status"] == "approved"
    info = describe(rows)
    assert info["n_variants"] == 300 and info["counts_by_ambiguity_type"]["clear"] == 100 and len(info["representative_examples"]) >= 10


def _results(with_probabilities: bool) -> pd.DataFrame:
    rows = []
    spec = {
        "s1": {"clear": ("a", 0.95), "competing_intents": ("a", 0.80), "underspecified": ("b", 0.55)},
        "s2": {"clear": ("a", 0.90), "competing_intents": ("b", 0.60), "underspecified": ("c", 0.40)},
    }
    for src, levels in spec.items():
        for level, (pred, conf) in levels.items():
            rows.append(
                {
                    "provider": "jev",
                    "source_example_id": src,
                    "ambiguity_type": level,
                    "prediction": pred,
                    "confidence": conf,
                    "correct": None,
                    "latency_ms": 1.0,
                    "primary_expected_intent": None if level == "underspecified" else "a",
                    "acceptable_intents": json.dumps(["a"] if level == "clear" else ["a", "b"]),
                    "probabilities_json": json.dumps({pred: conf, "z": 1 - conf}) if with_probabilities else None,
                }
            )
    return pd.DataFrame(rows)


def test_strict_versus_acceptable_set_accuracy_and_paired_flips() -> None:
    tables, summary = analyze(_results(with_probabilities=False))
    t = tables["ambiguity_by_condition"].set_index("ambiguity_type")
    assert t.loc["clear", "strict_accuracy"] == 1.0 and t.loc["clear", "acceptable_set_accuracy"] == 1.0
    assert t.loc["competing_intents", "strict_accuracy"] == 0.5  # only s1 matches the primary label 'a'
    assert t.loc["competing_intents", "acceptable_set_accuracy"] == 1.0  # 'b' is defensible for s2
    assert pd.isna(t.loc["underspecified", "strict_accuracy"]) and t.loc["underspecified", "n_strict_scorable"] == 0
    assert t.loc["underspecified", "acceptable_set_accuracy"] == 0.5  # s1 'b' ok, s2 'c' not
    assert t.loc["competing_intents", "decision_flip_rate_vs_clear"] == 0.5
    assert t.loc["competing_intents", "mean_confidence_delta_vs_clear"] == pytest.approx(((0.80 - 0.95) + (0.60 - 0.90)) / 2)
    assert t.loc["clear", "median_confidence"] == pytest.approx(0.925)
    pairs = tables["ambiguity_pairs"]
    assert set(pairs["source_example_id"]) == {"s1", "s2"} and "original_prediction" in pairs and "ambiguous_prediction" in pairs
    assert t.loc["underspecified", "n_distinct_predictions"] == 2 and t.loc["competing_intents", "modal_prediction_share"] == 0.5
    assert t.loc["underspecified", "acceptable_set_accuracy_role"] == "descriptive_only" and t.loc["clear", "strict_accuracy_role"] == "valid"
    assert set(summary["interpretation_rules"]) == {"clear", "competing_intents", "underspecified"}
    assert summary["ordinal_scale_claimed"] is False and "no monotonic" in summary["interpretation"]
    assert not any("monotonic" in c or "trend" in c for c in tables["ambiguity_by_condition"].columns)  # no cross-condition trend is computed


def test_margin_is_reported_only_when_probabilities_were_recorded() -> None:
    without = analyze(_results(False))[0]["ambiguity_by_condition"]
    assert without["mean_top1_top2_margin"].isna().all() and (without["n_with_margin"] == 0).all()
    assert without["margin_note"].str.startswith("unavailable").all()
    with_p = analyze(_results(True))[0]["ambiguity_by_condition"].set_index("ambiguity_type")
    assert with_p.loc["clear", "mean_top1_top2_margin"] == pytest.approx(((0.95 - 0.05) + (0.90 - 0.10)) / 2)
    assert enrich(_results(True))["margin"].notna().all()


def test_enrich_rejects_unknown_level() -> None:
    bad = _results(False)
    bad.loc[0, "ambiguity_type"] = "WEIRD"
    with pytest.raises(StressError):
        enrich(bad)
