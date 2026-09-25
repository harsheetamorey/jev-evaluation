"""Tests for deterministic context construction and paired context metrics. No network or API access."""

import pandas as pd
import pytest

from phase2.context_pollution import (
    CONTEXT_BUDGET_CHARS,
    LEVELS,
    analyze,
    build,
    build_history,
    describe,
    irrelevant_pool,
    leaks_label,
    rows_for_source,
)
from phase2.stress import StressError  # noqa: F401

POOL = [{"source_id": f"p{i}", "text": "x" * (10 + i)} for i in range(60)]


def test_context_construction_is_deterministic() -> None:
    assert build_history(POOL, "s1", 300) == build_history(POOL, "s1", 300)
    assert build_history(POOL, "s1", 300) != build_history(POOL, "s2", 300)  # different source, different shuffle
    assert build_history(POOL, "s1", 300, seed=1) != build_history(POOL, "s1", 300, seed=2)


def test_context_grows_with_level_and_levels_are_nested() -> None:
    sizes, previous = [], []
    for level, budget in CONTEXT_BUDGET_CHARS.items():
        ids = [f["source_id"] for f in build_history(POOL, "s1", budget)]
        assert ids[: len(previous)] == previous  # smaller level is a prefix of the larger
        sizes.append(sum(len(f["text"]) for f in build_history(POOL, "s1", budget)))
        assert sizes[-1] <= budget
        previous = ids
    assert sizes == sorted(sizes) and sizes[0] == 0 and len(set(sizes)) > 2


def test_fragments_that_name_a_candidate_label_are_skipped_not_used() -> None:
    pool = [{"source_id": "bad", "text": "please track order now"}, {"source_id": "ok1", "text": "play some jazz"}, {"source_id": "ok2", "text": "what is the weather"}]
    for src in ("a", "b", "c", "d"):
        ids = [f["source_id"] for f in build_history(pool, src, 1000, avoid_labels=["track_order", "get_refund"])]
        assert "bad" not in ids and set(ids) == {"ok1", "ok2"}
    assert "bad" in [f["source_id"] for f in build_history(pool, "a", 1000)]  # without the filter it can appear


def test_rows_track_sources_and_never_mutate_the_target_text() -> None:
    target = "i need my money back"
    rows = rows_for_source("bitext-1", target, ["get_refund", "track_refund"], "get_refund", POOL)
    assert [r["context_level"] for r in rows] == list(LEVELS)
    for r in rows:
        assert r["source_example_id"] == "bitext-1" and r["base_message"] == target == r["text"] == r["state_payload"]["message"]
        assert r["expected_label"] == "get_refund" and r["polluted_state"] == r["state_payload"]
        assert len(r["context_source_ids"]) == r["n_history_items"] == len(r["state_payload"].get("history", []))
        assert set(r["context_source_ids"]) <= {p["source_id"] for p in POOL}
    none = rows[0]
    assert none["state_payload"] == {"message": target} and none["context_source_ids"] == []  # base state has no history key
    assert [r["input_chars"] for r in rows] == sorted(r["input_chars"] for r in rows)


def test_context_never_contains_a_candidate_label() -> None:
    assert leaks_label(["please get refund now"], ["get_refund"])
    assert leaks_label(["GET_REFUND"], ["get_refund"])
    assert not leaks_label(["play some jazz"], ["get_refund", "track_order"])
    poisoned = [{"source_id": "p", "text": "check the track order status"}]
    rows = rows_for_source("s", "hi", ["track_order", "get_refund"], "track_order", poisoned, budgets={"small": 100})
    assert rows[0]["context_source_ids"] == []  # the only fragment names a label, so it is excluded rather than leaked


def test_built_dataset_is_deterministic_and_history_comes_from_the_unrelated_pool() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["n_source_examples"] == 100 and info["counts_by_context_level"] == {lvl: 100 for lvl in LEVELS}
    means = info["history_items_by_level_mean"]
    assert means["none"] == 0 and means["small"] < means["medium"] < means["large"] < means["near_limit"]
    pool_ids = {p["source_id"] for p in irrelevant_pool()}
    assert all(set(r["context_source_ids"]) <= pool_ids for r in rows)
    assert meta["irrelevant_pool_size"] == 250 and "NOT verified" in meta["context_limit_note"]
    assert len({r["variant_id"] for r in rows}) == len(rows) and len(info["representative_examples"]) == 3


def _results() -> pd.DataFrame:
    spec = {
        "s1": {"none": ("a", 0.9, True, 100, 50), "small": ("a", 0.85, True, 110, 90), "large": ("b", 0.5, False, 150, 400)},
        "s2": {"none": ("a", 0.9, True, 100, 50), "small": ("a", 0.9, True, 105, 90), "large": ("a", 0.7, True, 140, 400)},
    }
    rows = []
    for src, levels in spec.items():
        for level, (pred, conf, ok, lat, toks) in levels.items():
            rows.append({"provider": "jev", "source_example_id": src, "context_level": level, "expected_label": "a", "prediction": pred, "confidence": conf, "correct": ok, "latency_ms": float(lat), "input_tokens": toks, "input_chars": 10 * toks, "n_history_items": 0 if level == "none" else 3})
    return pd.DataFrame(rows)


def test_paired_context_metrics() -> None:
    tables, summary = analyze(_results())
    t = tables["context_by_level"].set_index("context_level")
    assert t.loc["small", "decision_flip_rate"] == 0.0 and t.loc["large", "decision_flip_rate"] == 0.5
    assert t.loc["large", "accuracy_delta"] == pytest.approx(-0.5) and t.loc["small", "accuracy_delta"] == 0.0
    assert t.loc["large", "mean_confidence_delta"] == pytest.approx(((0.5 - 0.9) + (0.7 - 0.9)) / 2)
    assert t.loc["large", "mean_latency_delta_ms"] == pytest.approx(((150 - 100) + (140 - 100)) / 2)
    assert t.loc["large", "mean_input_tokens_variant"] == 400 and t.loc["large", "n_pairs"] == 2
    assert list(t.index) == ["small", "large"]  # ordered by context size, not alphabetically
    sizes = tables["context_input_sizes"]
    assert set(sizes["context_level"]) == {"none", "small", "large"} and "not concluded" in summary["interpretation"]
    assert set(tables["context_pairs"]["source_example_id"]) == {"s1", "s2"}


def test_analyze_requires_no_context_rows() -> None:
    with pytest.raises(StressError, match="none"):
        analyze(_results().query("context_level != 'none'"))
