"""Tests for the ambiguity stress dataset and metrics. No network or API access."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.ambiguity import (
    FAMILIES,
    LEVELS,
    SIDE_QUESTION,
    acceptable_set_correct,
    analyze,
    build,
    build_variants,
    describe,
    enrich,
    pick_partner,
    shared_families,
    validate_level,
)
from phase2.stress import StressError, load_frozen_dataset, validate_rows, write_frozen_dataset


def test_ambiguity_level_validation() -> None:
    for level in LEVELS:
        assert validate_level(level) == level
    with pytest.raises(StressError, match="Unknown ambiguity_level"):
        validate_level("MILDLY_AMBIGUOUS")


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
    v = {r["ambiguity_level"]: r for r in build_variants("src-1", "i want my money back", ["get_refund", "track_refund", "place_order", "complaint", "review"], "get_refund", "track_refund", "refund")}
    assert {r["source_example_id"] for r in v.values()} == {"src-1"} and set(v) == set(LEVELS)
    assert v["CLEAR"]["text"] == "i want my money back" and v["CLEAR"]["acceptable_intents"] == ["get_refund"]
    assert v["SLIGHTLY_AMBIGUOUS"]["text"].startswith("i want my money back") and v["SLIGHTLY_AMBIGUOUS"]["primary_expected_intent"] == "get_refund"
    assert v["SLIGHTLY_AMBIGUOUS"]["acceptable_intents"] == ["get_refund", "track_refund"]
    high = v["HIGHLY_AMBIGUOUS"]
    assert high["ground_truth"] is None and high["primary_expected_intent"] is None
    assert high["acceptable_intents"] == ["get_refund", "track_refund"]  # family members present in the choice set (not check_refund_policy)
    assert all(r["original_text"] == "i want my money back" and r["original_ground_truth"] == "get_refund" for r in v.values())
    validate_rows(list(v.values()))


def test_acceptable_set_handling() -> None:
    assert acceptable_set_correct("a", ["a", "b"]) is True and acceptable_set_correct("c", ["a", "b"]) is False
    assert acceptable_set_correct(None, ["a"]) is None


def test_built_dataset_is_deterministic_complete_and_loads_from_a_frozen_dir(tmp_path: Path) -> None:
    rows, meta = build()
    rows2, _ = build()
    assert rows == rows2  # same bytes every time
    assert len(rows) == 300 and len({r["source_example_id"] for r in rows}) == 100
    counts = pd.Series([r["ambiguity_level"] for r in rows]).value_counts()
    assert set(counts) == {100} and set(counts.index) == set(LEVELS)
    for r in rows:  # ground truth is always among the choices; ambiguous items never invent a single truth
        assert set(r["acceptable_intents"]) <= set(r["candidates"]) and r["original_ground_truth"] in r["candidates"]
        assert (r["ground_truth"] is None) == (r["ambiguity_level"] == "HIGHLY_AMBIGUOUS")
    write_frozen_dataset(tmp_path, rows, meta)
    loaded, manifest = load_frozen_dataset(tmp_path)
    assert loaded == rows and manifest["n_rows"] == 300 and manifest["review_status"] == "pending_human_review"
    info = describe(rows)
    assert info["n_variants"] == 300 and info["counts_by_ambiguity_level"]["CLEAR"] == 100 and len(info["representative_examples"]) >= 10


def _results(with_probabilities: bool) -> pd.DataFrame:
    rows = []
    spec = {
        "s1": {"CLEAR": ("a", 0.95), "SLIGHTLY_AMBIGUOUS": ("a", 0.80), "HIGHLY_AMBIGUOUS": ("b", 0.55)},
        "s2": {"CLEAR": ("a", 0.90), "SLIGHTLY_AMBIGUOUS": ("b", 0.60), "HIGHLY_AMBIGUOUS": ("c", 0.40)},
    }
    for src, levels in spec.items():
        for level, (pred, conf) in levels.items():
            rows.append(
                {
                    "provider": "jev",
                    "source_example_id": src,
                    "ambiguity_level": level,
                    "prediction": pred,
                    "confidence": conf,
                    "correct": None,
                    "latency_ms": 1.0,
                    "primary_expected_intent": None if level == "HIGHLY_AMBIGUOUS" else "a",
                    "acceptable_intents": json.dumps(["a"] if level == "CLEAR" else ["a", "b"]),
                    "probabilities_json": json.dumps({pred: conf, "z": 1 - conf}) if with_probabilities else None,
                }
            )
    return pd.DataFrame(rows)


def test_strict_versus_acceptable_set_accuracy_and_paired_flips() -> None:
    tables, summary = analyze(_results(with_probabilities=False))
    t = tables["ambiguity_by_level"].set_index("ambiguity_level")
    assert t.loc["CLEAR", "strict_accuracy"] == 1.0 and t.loc["CLEAR", "acceptable_set_accuracy"] == 1.0
    assert t.loc["SLIGHTLY_AMBIGUOUS", "strict_accuracy"] == 0.5  # only s1 matches the primary label 'a'
    assert t.loc["SLIGHTLY_AMBIGUOUS", "acceptable_set_accuracy"] == 1.0  # 'b' is defensible for s2
    assert pd.isna(t.loc["HIGHLY_AMBIGUOUS", "strict_accuracy"]) and t.loc["HIGHLY_AMBIGUOUS", "n_strict_scorable"] == 0
    assert t.loc["HIGHLY_AMBIGUOUS", "acceptable_set_accuracy"] == 0.5  # s1 'b' ok, s2 'c' not
    assert t.loc["SLIGHTLY_AMBIGUOUS", "decision_flip_rate_vs_clear"] == 0.5
    assert t.loc["SLIGHTLY_AMBIGUOUS", "mean_confidence_delta_vs_clear"] == pytest.approx(((0.80 - 0.95) + (0.60 - 0.90)) / 2)
    assert t.loc["CLEAR", "median_confidence"] == pytest.approx(0.925)
    pairs = tables["ambiguity_pairs"]
    assert set(pairs["source_example_id"]) == {"s1", "s2"} and "original_prediction" in pairs and "ambiguous_prediction" in pairs
    assert "not concluded" in summary["interpretation"]


def test_margin_is_reported_only_when_probabilities_were_recorded() -> None:
    without = analyze(_results(False))[0]["ambiguity_by_level"]
    assert without["mean_top1_top2_margin"].isna().all() and (without["n_with_margin"] == 0).all()
    assert without["margin_note"].str.startswith("unavailable").all()
    with_p = analyze(_results(True))[0]["ambiguity_by_level"].set_index("ambiguity_level")
    assert with_p.loc["CLEAR", "mean_top1_top2_margin"] == pytest.approx(((0.95 - 0.05) + (0.90 - 0.10)) / 2)
    assert enrich(_results(True))["margin"].notna().all()


def test_enrich_rejects_unknown_level() -> None:
    bad = _results(False)
    bad.loc[0, "ambiguity_level"] = "WEIRD"
    with pytest.raises(StressError):
        enrich(bad)
