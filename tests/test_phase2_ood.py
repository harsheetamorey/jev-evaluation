"""Tests for the OOD dataset, contracts and metrics. No network or API access."""

import pandas as pd
import pytest

from phase2.ood import (
    FALSE_CONFIDENCE_THRESHOLDS,
    OTHER_LABEL,
    analyze,
    build,
    contract_candidates,
    describe,
    false_confidence_rates,
    is_support_related,
    ood_pool,
    rows_for_pair,
)
from phase2.stress import StressError, select_bitext_sources

CHOICES = ["get_refund", "track_refund", "cancel_order", "place_order", "complaint"]


def test_forced_choice_and_explicit_fallback_differ_only_by_the_other_label() -> None:
    forced, fallback = contract_candidates(CHOICES, "forced_choice"), contract_candidates(CHOICES, "explicit_fallback")
    assert forced == CHOICES and fallback == [*CHOICES, OTHER_LABEL] and OTHER_LABEL not in forced
    with pytest.raises(StressError):
        contract_candidates(CHOICES, "made_up")
    ood = {"source_id": "massive:en-US:1", "text": "play some jazz please", "massive_intent": "play_music"}
    rows = rows_for_pair("bitext-1", "i want my money back", "get_refund", CHOICES, ood)
    by = {(r["population"], r["contract_type"]): r for r in rows}
    assert len(rows) == 4
    for pop in ("in_domain", "ood"):
        assert by[(pop, "forced_choice")]["text"] == by[(pop, "explicit_fallback")]["text"]  # same input under both contracts
        assert by[(pop, "explicit_fallback")]["candidates"] == by[(pop, "forced_choice")]["candidates"] + [OTHER_LABEL]
    assert by[("ood", "forced_choice")]["ground_truth"] is None  # no correct choice exists
    assert by[("in_domain", "forced_choice")]["ground_truth"] == "get_refund" == by[("in_domain", "explicit_fallback")]["ground_truth"]
    assert by[("ood", "explicit_fallback")]["ground_truth"] == OTHER_LABEL
    assert by[("ood", "forced_choice")]["candidates"] == by[("in_domain", "forced_choice")]["candidates"]  # matched candidate set


def test_ood_pool_contains_no_support_language() -> None:
    assert is_support_related("please track my order") and is_support_related("i want a refund") and not is_support_related("play some jazz")
    pool = ood_pool()
    assert len(pool) >= 100 and not any(is_support_related(p["text"]) for p in pool)


def test_built_dataset_keeps_populations_separate_and_persists_exact_ids() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["counts_by_population_and_contract"] == {f"{p}/{c}": 100 for p in ("in_domain", "ood") for c in ("explicit_fallback", "forced_choice")}
    bitext_ids = {r.id for r in select_bitext_sources(1000)}
    ood_ids = {r["source_example_id"] for r in rows if r["population"] == "ood"}
    control_ids = {r["source_example_id"] for r in rows if r["population"] == "in_domain"}
    assert ood_ids.isdisjoint(bitext_ids) and control_ids <= bitext_ids and len(ood_ids) == len(control_ids) == 100
    assert set(meta["ood_source_ids"]) == ood_ids and set(meta["control_source_ids"]) == control_ids  # exact IDs persisted
    assert meta["ood_source_file_sha256"] and "not_recorded" in meta["ood_dataset_revision"] and len(info["ood_examples"]) == 10
    assert len({r["variant_id"] for r in rows}) == 400


def test_false_confidence_thresholds() -> None:
    conf = pd.Series([0.99, 0.95, 0.90, 0.85, 0.80, 0.50, None])
    rates = false_confidence_rates(conf)
    assert set(rates) == {f"share_conf_ge_{t:.2f}" for t in FALSE_CONFIDENCE_THRESHOLDS}
    assert rates["share_conf_ge_0.80"] == pytest.approx(5 / 6) and rates["share_conf_ge_0.90"] == pytest.approx(3 / 6)
    assert rates["share_conf_ge_0.95"] == pytest.approx(2 / 6)  # thresholds are inclusive
    assert false_confidence_rates(pd.Series([], dtype=float))["share_conf_ge_0.90"] is None


def _results() -> pd.DataFrame:
    rows = []

    def add(pop, contract, pred, conf, correct=None):
        rows.append({"provider": "jev", "population": pop, "contract_type": contract, "prediction": pred, "confidence": conf, "correct": correct})

    for pred, conf in [("get_refund", 0.95), ("get_refund", 0.90), ("place_order", 0.6), ("complaint", 0.4)]:
        add("ood", "forced_choice", pred, conf)
    for pred, conf in [("other", 0.7), ("other", 0.8), ("get_refund", 0.9), ("place_order", 0.5)]:
        add("ood", "explicit_fallback", pred, conf, pred == "other")
    for pred, conf, ok in [("get_refund", 0.99, True), ("track_refund", 0.97, True), ("cancel_order", 0.9, False), ("get_refund", 0.98, True)]:
        add("in_domain", "forced_choice", pred, conf, ok)
    for pred, conf, ok in [("get_refund", 0.99, True), ("other", 0.6, False), ("cancel_order", 0.9, True), ("get_refund", 0.98, True)]:
        add("in_domain", "explicit_fallback", pred, conf, ok)
    return pd.DataFrame(rows)


def test_fallback_rate_confidence_split_and_forced_choice_metrics() -> None:
    tables, summary = analyze(_results())
    fb = tables["ood_fallback"].set_index("population")
    assert fb.loc["ood", "other_selection_rate"] == 0.5 and fb.loc["ood", "non_other_rate"] == 0.5 and fb.loc["ood", "n"] == 4
    assert fb.loc["ood", "mean_confidence_when_other"] == pytest.approx(0.75)
    assert fb.loc["ood", "mean_confidence_when_forced_into_in_domain_intent"] == pytest.approx(0.7)
    assert fb.loc["in_domain", "other_selection_rate"] == 0.25 and fb.loc["ood", "non_other_share_conf_ge_0.90"] == pytest.approx(1 / 4)
    forced = tables["ood_forced_choice"].set_index("population")
    assert forced.loc["ood", "share_conf_ge_0.90"] == 0.5 and forced.loc["ood", "share_conf_ge_0.95"] == 0.25
    assert forced.loc["in_domain", "accuracy"] == 0.75
    assert pd.isna(forced.loc["ood", "accuracy"])  # no accuracy is invented for OOD, where no choice is correct
    labels = tables["ood_label_distribution"]
    assert labels[(labels["population"] == "ood")].set_index("predicted_label").loc["get_refund", "count"] == 2
    cmp = tables["ood_confidence_in_domain_vs_ood"].set_index("contract_type")
    assert cmp.loc["forced_choice", "n_in_domain"] == 4 and cmp.loc["forced_choice", "mean_conf_ood"] == pytest.approx(0.7125)
    assert 0 <= cmp.loc["forced_choice", "ks_statistic_conf_in_domain_vs_ood"] <= 1
    assert "not evidence that Jev detects OOD" in summary["caveat"] and "not concluded" in summary["interpretation"]


def test_confidence_bins_include_every_bin_and_the_top_bin_is_closed() -> None:
    bins = analyze(_results())[0]["ood_confidence_bins"]
    one = bins[(bins["population"] == "ood") & (bins["contract_type"] == "forced_choice")]
    assert len(one) == 10 and one["sample_count"].sum() == 4 and one.iloc[-1]["bin_label"] == "[0.9, 1.0]" and one.iloc[-1]["sample_count"] == 2


def test_analyze_rejects_missing_columns() -> None:
    with pytest.raises(StressError, match="missing columns"):
        analyze(_results().drop(columns=["contract_type"]))
