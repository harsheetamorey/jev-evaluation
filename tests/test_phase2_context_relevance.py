"""Tests for relevance triplets, paired joins and transition metrics. No network or API access."""

import pandas as pd
import pytest

from phase2.context_relevance import (
    CONDITIONS,
    TRIPLETS,
    analyze,
    build,
    candidates_by_intent,
    describe,
    rows_for_triplet,
    transition_counts,
    validate_triplet,
)
from phase2.stress import StressError, validate_rows

CANDS = ["payment_issue", "check_payment_methods", "get_refund", "check_invoice", "get_invoice"]
GOOD = ("payment_issue", "It happened again.", "Earlier the customer said their card payment was declined at checkout.", "Earlier the customer said they prefer to be contacted in the afternoon.", ["declined"])


def test_every_authored_triplet_passes_validation() -> None:
    cand = candidates_by_intent()
    assert len(TRIPLETS) == 30
    for label, target, rel, irr, cues in TRIPLETS:
        assert validate_triplet(label, target, rel, irr, cues, cand[label]) == [], (label, target)


def test_validator_rejects_unsound_triplets() -> None:
    label, target, rel, irr, cues = GOOD
    assert validate_triplet(label, target, rel, irr, cues, CANDS) == []
    assert any("missing from relevant" in p for p in validate_triplet(label, target, "Earlier the customer said hello there friend.", irr, cues, CANDS))
    assert any("appears in irrelevant" in p for p in validate_triplet(label, target, rel, "Earlier the customer said the payment was declined somewhere.", cues, CANDS))
    assert any("appears in the target" in p for p in validate_triplet(label, "It was declined again.", rel, irr, cues, CANDS))
    assert any("names other candidates" in p for p in validate_triplet(label, target, "Earlier the customer said the card payment was declined and wants a refund.", irr, cues, CANDS))
    assert any("irrelevant history names" in p for p in validate_triplet(label, target, rel, "Earlier the customer asked about a refund on something else.", cues, CANDS))
    assert any("not vague" in p for p in validate_triplet(label, "I want a refund now.", rel, irr, cues, CANDS))
    assert any("not comparable" in p for p in validate_triplet(label, target, rel, "Hi there.", cues, CANDS))
    assert any("not among candidates" in p for p in validate_triplet("cancel_order", target, rel, irr, cues, CANDS))
    assert any("no relevance cues" in p for p in validate_triplet(label, target, rel, irr, [], CANDS))


def test_triplet_rows_share_source_target_and_label_and_differ_only_in_history() -> None:
    rows = rows_for_triplet(7, *GOOD, CANDS)
    assert [r["context_condition"] for r in rows] == list(CONDITIONS)
    assert {r["source_example_id"] for r in rows} == {"triplet-007"} and {r["text"] for r in rows} == {GOOD[1]}
    assert {r["expected_label"] for r in rows} == {"payment_issue"} and all(r["candidates"] == CANDS for r in rows)
    none, irr, rel = (r["state_payload"] for r in rows)
    assert none == {"message": GOOD[1]} and irr["history"] == [GOOD[3]] and rel["history"] == [GOOD[2]]
    validate_rows(rows)
    with pytest.raises(StressError, match="failed validation"):
        rows_for_triplet(8, GOOD[0], GOOD[1], "nothing useful here at all", GOOD[3], GOOD[4], CANDS)


def test_built_dataset_is_deterministic_and_complete() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["n_triplets"] == 30 and info["n_rows"] == 90 and info["counts_by_condition"] == {c: 30 for c in CONDITIONS}
    assert info["distinct_expected_intents"] >= 20 and len(info["example_triplets"]) >= 10
    assert meta["review_status"] == "pending_human_review" and "no model generation" in meta["generation_method"]
    assert len({r["variant_id"] for r in rows}) == 90


def _results() -> pd.DataFrame:
    # triplet -> condition -> (prediction, confidence, correct)
    spec = {
        "t1": {"no_context": ("b", 0.4, False), "irrelevant_context": ("b", 0.4, False), "relevant_context": ("a", 0.9, True)},  # wrong -> correct with relevant
        "t2": {"no_context": ("a", 0.8, True), "irrelevant_context": ("c", 0.5, False), "relevant_context": ("a", 0.9, True)},  # correct -> wrong with irrelevant
        "t3": {"no_context": ("a", 0.7, True), "irrelevant_context": ("a", 0.7, True), "relevant_context": ("b", 0.6, False)},  # correct -> wrong with relevant
        "t4": {"no_context": ("c", 0.5, False), "irrelevant_context": ("a", 0.6, True), "relevant_context": ("c", 0.5, False)},  # wrong -> correct with irrelevant
    }
    rows = []
    for t, conds in spec.items():
        for cond, (pred, conf, ok) in conds.items():
            rows.append({"provider": "jev", "source_example_id": t, "context_condition": cond, "expected_label": "a", "prediction": pred, "confidence": conf, "correct": ok, "latency_ms": 1.0})
    return pd.DataFrame(rows)


def test_paired_transitions_are_counted_in_the_right_direction() -> None:
    tables, summary = analyze(_results())
    t = tables["relevance_transitions"].set_index("comparison")
    rel, irr = t.loc["relevant_context vs no_context"], t.loc["irrelevant_context vs no_context"]
    assert (rel["wrong_to_correct"], rel["correct_to_wrong"], rel["correct_to_correct"], rel["wrong_to_wrong"]) == (1, 1, 1, 1)
    assert (irr["wrong_to_correct"], irr["correct_to_wrong"], irr["correct_to_correct"], irr["wrong_to_wrong"]) == (1, 1, 1, 1)
    direct = t.loc["relevant_context vs irrelevant_context"]  # base = irrelevant
    assert (direct["wrong_to_correct"], direct["correct_to_wrong"]) == (2, 2)  # t1,t2 improve; t3,t4 degrade
    assert direct["n_scored_pairs"] == 4 and "not concluded" in summary["interpretation"]


def test_condition_metrics_and_paired_flips() -> None:
    by = analyze(_results())[0]["relevance_by_condition"].set_index("context_condition")
    assert by.loc["no_context", "accuracy"] == 0.5 and by.loc["relevant_context", "accuracy"] == 0.5 and by.loc["irrelevant_context", "accuracy"] == 0.5
    assert by.loc["relevant_context", "decision_flips_vs_no_context"] == 0.5  # t1 and t3 changed
    assert by.loc["irrelevant_context", "decision_flips_vs_no_context"] == 0.5  # t2 and t4 changed
    assert by.loc["relevant_context", "mean_confidence_delta_vs_no_context"] == pytest.approx((0.5 + 0.1 - 0.1 + 0.0) / 4)
    assert by.loc["no_context", "n"] == 4


def test_transition_counts_skip_unscored_pairs() -> None:
    paired = pd.DataFrame({"correct_base": [True, None, False], "correct_variant": [False, True, None]})
    c = transition_counts(paired)
    assert c["n_scored_pairs"] == 1 and c["correct_to_wrong"] == 1 and c["wrong_to_correct"] == 0


def test_analyze_requires_no_context_rows() -> None:
    with pytest.raises(StressError, match="no_context"):
        analyze(_results().query("context_condition != 'no_context'"))
