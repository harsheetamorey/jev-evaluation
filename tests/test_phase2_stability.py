"""Tests for the semantic-stability dataset and paired metrics. No network or API access."""

import pandas as pd
import pytest

from phase2.stability import (
    PARAPHRASES,
    SURFACE_FUNCS,
    analyze,
    build,
    confidence_variance_within_families,
    describe,
    detect_duplicate_variants,
    surface_compress,
    surface_exclaim,
    surface_lowercase_nopunct,
    surface_typo,
    variants_for_source,
)
from phase2.stress import StressError

BITEXT_INTENTS = 27


def test_surface_transforms_are_deterministic_pure_and_do_what_they_say() -> None:
    text = "I want to cancel my order, please!"
    before = text
    for name, fn in SURFACE_FUNCS.items():
        assert fn(text, "src-1") == fn(text, "src-1"), name
    assert text == before  # no mutation
    assert SURFACE_FUNCS["surface_uppercase"](text, "s") == "I WANT TO CANCEL MY ORDER, PLEASE!"
    assert surface_exclaim(text, "s") == "I want to cancel my order, please!!!"
    assert surface_lowercase_nopunct(text, "s") == "i want to cancel my order please"
    assert surface_compress(text, "s") == "cancel my order"
    typo = surface_typo(text, "src-1")
    assert len(typo) == len(text) - 1 and typo != text
    assert surface_typo("ok", "s") == "ok"  # nothing long enough to corrupt


def test_typo_position_depends_only_on_source_id() -> None:
    a = surface_typo("i want to cancel my order", "same-id")
    assert a == surface_typo("i want to cancel my order", "same-id")
    positions = {surface_typo("i want to cancel my order", f"id-{i}") for i in range(20)}
    assert len(positions) > 1  # different sources get different typo positions


def test_paraphrase_bank_covers_every_intent_and_never_leaks_a_snake_case_label() -> None:
    assert len(PARAPHRASES) == BITEXT_INTENTS and all(len(v) == 3 for v in PARAPHRASES.values())
    for intent, texts in PARAPHRASES.items():
        assert len(set(texts)) == 3 and all("_" not in t for t in texts)  # natural language, never a raw label like get_refund


def test_duplicate_variant_detection() -> None:
    d = detect_duplicate_variants("Cancel my order", {"a": "Cancel  my order ", "b": "Something new", "c": "Something   new", "d": "CANCEL MY ORDER"})
    assert d == {"a": "duplicate of original", "c": "duplicate of b"}  # whitespace-only; a case change is a real variant, not a duplicate


def test_variants_retain_source_id_and_drop_duplicates() -> None:
    rows, dropped = variants_for_source("src-1", "CANCEL", ["cancel_order", "track_order"], "cancel_order")
    assert {r["source_example_id"] for r in rows} == {"src-1"} and rows[0]["variant_type"] == "original"
    assert "surface_uppercase" in dropped  # already upper case -> identical to the original
    assert all(r["original_text"] == "CANCEL" and r["expected_label"] == "cancel_order" for r in rows)
    assert len({r["variant_id"] for r in rows}) == len(rows)
    assert {r["variant_family"] for r in rows} <= {"original", "surface", "paraphrase"}


def test_built_dataset_is_deterministic_and_families_are_separate() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["n_source_examples"] == 100 and info["counts_by_variant_type"]["original"] == 100
    assert info["counts_by_family"]["paraphrase"] == 300 and len(info["example_families"]) == 10
    assert meta["review_status"] == "pending_human_review" and "no model generation" in meta["generation_method"]
    for r in rows:
        assert (r["variant_family"] == "paraphrase") == r["variant_type"].startswith("paraphrase_")
        assert r["expected_label"] in r["candidates"]
    by_src = pd.DataFrame(rows).groupby("source_example_id")["text"].apply(lambda s: s.str.strip().str.replace(r"\s+", " ", regex=True).nunique() == len(s))
    assert by_src.all()  # no duplicate texts survive within a source
    assert info["counts_by_variant_type"]["surface_uppercase"] > 90  # uppercase variants are kept (only already-uppercase sources would collide)


def _results() -> pd.DataFrame:
    rows = []
    spec = {
        "s1": {"original": ("a", 0.9, True), "surface_typo": ("a", 0.8, True), "paraphrase_1": ("b", 0.5, False)},
        "s2": {"original": ("a", 0.9, True), "surface_typo": ("a", 0.9, True), "paraphrase_1": ("a", 0.7, True)},
    }
    for src, vs in spec.items():
        for vtype, (pred, conf, ok) in vs.items():
            rows.append({"provider": "jev", "source_example_id": src, "variant_type": vtype, "variant_family": vtype.split("_")[0], "intent": "cancel_order", "expected_label": "a", "prediction": pred, "confidence": conf, "correct": ok, "latency_ms": 1.0})
    return pd.DataFrame(rows)


def test_pair_joining_label_agreement_flip_rate_and_confidence_delta() -> None:
    tables, summary = analyze(_results())
    t = tables["stability_by_variant_type"].set_index("variant_type")
    assert t.loc["surface_typo", "decision_flip_rate"] == 0.0 and t.loc["surface_typo", "label_agreement"] == 1.0
    assert t.loc["paraphrase_1", "decision_flip_rate"] == 0.5 and t.loc["paraphrase_1", "n_pairs"] == 2
    assert t.loc["paraphrase_1", "accuracy_base"] == 1.0 and t.loc["paraphrase_1", "accuracy_variant"] == 0.5
    assert t.loc["paraphrase_1", "accuracy_delta"] == pytest.approx(-0.5)
    assert t.loc["surface_typo", "mean_abs_confidence_delta"] == pytest.approx((0.1 + 0.0) / 2)
    assert t.loc["paraphrase_1", "mean_confidence_delta"] == pytest.approx(((0.5 - 0.9) + (0.7 - 0.9)) / 2)
    pairs = tables["stability_pairs"]
    assert set(pairs["source_example_id"]) == {"s1", "s2"} and (pairs["confidence_delta"].notna()).all()
    fam = tables["stability_by_family"].set_index("variant_family")
    assert set(fam.index) == {"surface", "paraphrase"} and summary["pairing"].startswith("every variant")


def test_confidence_variance_within_a_source_family() -> None:
    v = confidence_variance_within_families(_results()).set_index("source_example_id")
    assert v.loc["s1", "confidence_variance"] == pytest.approx(pd.Series([0.9, 0.8, 0.5]).var(ddof=0))
    assert v.loc["s1", "n_distinct_predictions"] == 2 and v.loc["s2", "n_distinct_predictions"] == 1


def test_analyze_requires_originals() -> None:
    with pytest.raises(StressError, match="original"):
        analyze(_results().query("variant_type != 'original'"))
