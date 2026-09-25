"""Tests for choice-overlap construction and paired metrics. No network or API access."""

import json

import pandas as pd
import pytest

from phase2.choice_overlap import (
    FAMILIES,
    FAMILY_OF,
    K_VALUES,
    SET_TYPES,
    analyze,
    build,
    build_set,
    describe,
    distinct_set,
    family_of,
    overlapping_set,
    rows_for_source,
    validate_choice_set,
)
from phase2.stress import StressError


def test_every_bitext_intent_has_exactly_one_family() -> None:
    all_intents = [i for members in FAMILIES.values() for i in members]
    assert len(all_intents) == len(set(all_intents)) == 27
    with pytest.raises(StressError):
        family_of("not_an_intent")


@pytest.mark.parametrize("k", K_VALUES)
def test_ground_truth_is_always_included_and_sets_have_exactly_k_unique_labels(k: int) -> None:
    for i in range(60):
        truth = ["cancel_order", "create_account", "delivery_options"][i % 3]
        for set_type in SET_TYPES:
            labels = build_set(f"src-{i}", truth, k, set_type)
            assert truth in labels and len(labels) == len(set(labels)) == k
            validate_choice_set(truth, labels, k, set_type)


def test_overlapping_sets_stay_in_the_truths_family_and_distinct_sets_leave_it() -> None:
    for i in range(60):
        truth = "create_account"
        over, dist = overlapping_set(f"s{i}", truth, 4), distinct_set(f"s{i}", truth, 4)
        assert {family_of(x) for x in over} == {"account"}
        assert all(family_of(x) != "account" for x in dist if x != truth)
        assert len({family_of(x) for x in dist}) == 4  # every label from a different family
    with pytest.raises(StressError, match="too small"):
        overlapping_set("s", "review", 3)  # singleton family cannot form an overlapping set


def test_validation_rejects_bad_sets() -> None:
    with pytest.raises(StressError):
        validate_choice_set("get_refund", ["get_refund", "track_refund", "cancel_order"], 3, "overlapping")  # cancel_order not in family
    with pytest.raises(StressError):
        validate_choice_set("get_refund", ["get_refund", "track_refund", "place_order"], 3, "distinct")  # track_refund shares the family
    with pytest.raises(StressError):
        validate_choice_set("get_refund", ["track_refund", "check_refund_policy", "cancel_order"], 3, "overlapping")  # truth missing


def test_choice_sets_are_reproducible_but_vary_across_sources() -> None:
    assert build_set("src-1", "cancel_order", 4, "overlapping") == build_set("src-1", "cancel_order", 4, "overlapping")
    assert len({tuple(build_set(f"src-{i}", "cancel_order", 4, "overlapping")) for i in range(30)}) > 1
    assert len({tuple(sorted(build_set(f"src-{i}", "cancel_order", 3, "distinct"))) for i in range(30)}) > 1


def test_rows_pair_the_same_source_across_both_set_types_at_each_k() -> None:
    rows = rows_for_source("bitext-9", "i want to cancel", "cancel_order")
    assert len(rows) == 4 and {r["source_example_id"] for r in rows} == {"bitext-9"} and {r["text"] for r in rows} == {"i want to cancel"}
    assert {(r["number_of_choices"], r["choice_set_type"]) for r in rows} == {(k, t) for k in K_VALUES for t in SET_TYPES}
    assert all(r["ground_truth"] in r["candidates"] and len(r["candidates"]) == r["number_of_choices"] for r in rows)
    assert len({r["choice_set_id"] for r in rows}) == 4


def test_built_dataset_is_deterministic_and_covers_pairs() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["n_source_examples"] == 100 and info["n_rows"] == 400 and info["n_choice_sets"] == 400
    assert info["counts_by_k_and_type"] == {f"k{k}/{t}": 100 for k in K_VALUES for t in SET_TYPES}
    assert set(info["sources_by_family"]) <= {f for f, m in FAMILIES.items() if len(m) >= max(K_VALUES)}
    assert all(FAMILY_OF[r["ground_truth"]] == r["family"] for r in rows) and len(info["example_candidate_sets"]) == 8
    assert "no model" in meta["generation_method"]


def _results(with_probabilities: bool) -> pd.DataFrame:
    rows = []
    spec = {
        "s1": {("distinct", 4): ("a", 0.95, True), ("overlapping", 4): ("b", 0.55, False)},
        "s2": {("distinct", 4): ("a", 0.90, True), ("overlapping", 4): ("a", 0.70, True)},
    }
    for src, cells in spec.items():
        for (stype, k), (pred, conf, ok) in cells.items():
            rows.append({"provider": "jev", "source_example_id": src, "choice_set_type": stype, "number_of_choices": k, "expected_label": "a", "prediction": pred, "confidence": conf, "correct": ok, "latency_ms": 1.0, "probabilities_json": json.dumps({pred: conf, "z": 1 - conf}) if with_probabilities else None})
    return pd.DataFrame(rows)


def test_paired_comparison_accuracy_difference_flips_and_confidence() -> None:
    tables, summary = analyze(_results(False))
    p = tables["choice_overlap_paired"].iloc[0]
    assert p["n_pairs"] == 2 and p["accuracy_distinct"] == 1.0 and p["accuracy_overlapping"] == 0.5
    assert p["accuracy_difference_overlapping_minus_distinct"] == pytest.approx(-0.5) and p["decision_flip_rate"] == 0.5
    assert p["mean_confidence_delta_overlapping_minus_distinct"] == pytest.approx(((0.55 - 0.95) + (0.70 - 0.90)) / 2)
    by = tables["choice_overlap_by_set"].set_index("choice_set_type")
    assert by.loc["distinct", "accuracy"] == 1.0 and by.loc["overlapping", "n"] == 2
    assert set(tables["choice_overlap_pairs"]["source_example_id"]) == {"s1", "s2"} and "not concluded" in summary["interpretation"]


def test_margin_only_when_full_probabilities_exist() -> None:
    without = analyze(_results(False))[0]["choice_overlap_by_set"]
    assert without["mean_top1_top2_margin"].isna().all() and (without["n_with_margin"] == 0).all() and without["margin_note"].str.startswith("unavailable").all()
    with_p = analyze(_results(True))[0]["choice_overlap_by_set"].set_index("choice_set_type")
    assert with_p.loc["distinct", "mean_top1_top2_margin"] == pytest.approx(((0.95 - 0.05) + (0.90 - 0.10)) / 2)
    assert with_p.loc["overlapping", "n_with_margin"] == 2
