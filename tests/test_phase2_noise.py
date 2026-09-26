"""Tests for deterministic noise transformations and paired noise metrics. No network or API access."""

import difflib

import pandas as pd
import pytest

from phase2.noise import (
    NOISE_FUNCS,
    NOISE_SEED,
    SEVERITIES,
    analyze,
    apply_noise,
    build,
    describe,
    rows_for_source,
    semantic_integrity,
)
from phase2.stress import StressError, select_bitext_sources

TEXT = "Please help me cancel my order because I do not need the payment information anymore."


def test_same_source_seed_type_and_severity_give_identical_output_every_time() -> None:
    for nt in NOISE_FUNCS:
        for sev in SEVERITIES:
            assert apply_noise(TEXT, "src-1", nt, sev) == apply_noise(TEXT, "src-1", nt, sev), (nt, sev)


def test_seed_and_source_id_change_the_randomised_types_but_severity_zero_is_identity() -> None:
    assert apply_noise(TEXT, "src-1", "typos", 3, seed=1) != apply_noise(TEXT, "src-1", "typos", 3, seed=2)
    assert len({apply_noise(TEXT, f"src-{i}", "typos", 2) for i in range(10)}) > 1
    for nt in NOISE_FUNCS:
        assert apply_noise(TEXT, "src-1", nt, 0) == TEXT


def test_originals_are_not_mutated_and_source_linkage_is_kept() -> None:
    sources = select_bitext_sources(5)
    snapshot = [(r.id, r.text) for r in sources]
    rows, _ = rows_for_source(sources[0].id, sources[0].text, sources[0].choices, sources[0].ground_truth)
    assert [(r.id, r.text) for r in sources] == snapshot
    assert all(r["source_example_id"] == sources[0].id and r["original_text"] == sources[0].text for r in rows)
    assert rows[0]["severity"] == 0 and rows[0]["text"] == sources[0].text and rows[0]["noise_type"] == "none"
    for r in rows:
        assert {"noise_type", "severity", "transformed_text", "expected_label", "seed"} <= set(r) and r["seed"] == NOISE_SEED


def _avg_distance(nt: str, sev: int) -> float:
    total = 0.0
    for i in range(40):
        out = apply_noise(TEXT, f"src-{i}", nt, sev)
        total += 1 - difflib.SequenceMatcher(None, TEXT, out).ratio()
    return total / 40


@pytest.mark.parametrize("noise_type", ["typos", "whitespace", "emoji", "abbreviations", "word_duplication", "extra_punctuation", "capitalization"])
def test_heavier_severity_changes_the_text_at_least_as_much(noise_type: str) -> None:
    s1, s2, s3 = (_avg_distance(noise_type, s) for s in SEVERITIES)
    assert 0 < s1 <= s2 + 1e-9 <= s3 + 2e-9, (noise_type, s1, s2, s3)


def test_missing_punctuation_removes_progressively_and_protects_placeholders() -> None:
    text = "Hello, could you help me, please? Order {{Order Number}}!"
    s1, s2, s3 = (apply_noise(text, "s", "missing_punctuation", sev) for sev in SEVERITIES)
    assert s1 == "Hello, could you help me, please? Order {{Order Number}}"
    assert s3 == "Hello could you help me please Order {{Order Number}}" and "{{Order Number}}" in s2
    count = lambda t: sum(c in ",?!" for c in t)  # noqa: E731
    assert count(text) > count(s1) >= count(s2) >= count(s3)


def test_specific_transformations() -> None:
    assert apply_noise("track my order", "s", "abbreviations", 1) == "track my order"  # nothing abbreviable
    assert apply_noise("please help you", "s", "abbreviations", 3) == "pls help u"
    assert "  " in apply_noise("cancel my order now", "s", "whitespace", 2)
    assert apply_noise("cancel my order", "s", "emoji", 1)[-1] in "🙂😡🙏😅👍❗"
    assert len(apply_noise("cancel my order", "s", "word_duplication", 1).split()) == 4
    with pytest.raises(StressError):
        apply_noise("x", "s", "made_up", 1)
    with pytest.raises(StressError):
        apply_noise("x y z", "s", "typos", 4)


def test_semantic_integrity_flags_unreadable_text_without_dropping_it() -> None:
    assert semantic_integrity(TEXT, TEXT + " 🙂🙂")[0] == "valid"
    assert semantic_integrity("cancel my order", "cancel my ordr")[0] == "valid"
    verdict, ratio, recall = semantic_integrity("cancel my order", "xq zzk lm")
    assert verdict == "questionable" and ratio < 0.7 and recall == 0.0


def test_unchanged_variants_are_skipped_and_counted_not_duplicated() -> None:
    rows, skipped = rows_for_source("s", "ok", ["a", "b"], "a")  # too short for most noise
    texts = [r["text"] for r in rows]
    assert skipped > 0 and len(texts) == len(set(texts))


def test_built_dataset_is_deterministic_with_all_types_and_severities() -> None:
    rows, meta = build()
    assert rows == build()[0]
    info = describe(rows)
    assert info["n_source_examples"] == 100 and set(info["counts_by_noise_type"]) == set(NOISE_FUNCS)
    assert set(info["counts_by_severity"]) == {1, 2, 3} and meta["seed"] == NOISE_SEED
    assert len({r["variant_id"] for r in rows}) == len(rows)
    assert info["semantic_integrity_counts"].get("questionable", 0) > 0  # heavy noise does produce flagged rows


def _results() -> pd.DataFrame:
    spec = {
        "s1": {(0, "none", "valid"): ("a", 0.9, True), (1, "typos", "valid"): ("a", 0.8, True), (3, "typos", "questionable"): ("b", 0.4, False)},
        "s2": {(0, "none", "valid"): ("a", 0.9, True), (1, "typos", "valid"): ("b", 0.6, False), (3, "typos", "valid"): ("a", 0.7, True)},
    }
    rows = []
    for src, variants in spec.items():
        for (sev, nt, integ), (pred, conf, ok) in variants.items():
            rows.append({"provider": "jev", "source_example_id": src, "severity": sev, "noise_type": nt, "semantic_integrity": integ, "intent": "x", "expected_label": "a", "prediction": pred, "confidence": conf, "correct": ok, "latency_ms": 1.0})
    return pd.DataFrame(rows)


def test_paired_noise_metrics_flip_rate_drop_delta_and_integrity_split() -> None:
    tables, summary = analyze(_results())
    sev = tables["noise_by_severity"].set_index(["severity", "semantic_integrity"])
    assert sev.loc[(1, "valid"), "n_pairs"] == 2 and sev.loc[(1, "valid"), "decision_flip_rate"] == 0.5
    assert sev.loc[(1, "valid"), "accuracy_base"] == 1.0 and sev.loc[(1, "valid"), "accuracy_variant"] == 0.5
    assert sev.loc[(1, "valid"), "accuracy_drop"] == pytest.approx(0.5)
    assert sev.loc[(1, "valid"), "mean_confidence_delta"] == pytest.approx(((0.8 - 0.9) + (0.6 - 0.9)) / 2)
    assert (3, "questionable") in sev.index and (3, "valid") in sev.index  # questionable kept separate, not merged into failures
    valid_only = tables["noise_valid_only_by_severity"].set_index("severity")
    assert valid_only.loc[3, "n_pairs"] == 1 and valid_only.loc[3, "decision_flip_rate"] == 0.0
    assert summary["n_pairs_semantic_integrity_questionable"] == 1 and set(tables["noise_pairs"]["source_example_id"]) == {"s1", "s2"}


def test_analyze_requires_clean_rows() -> None:
    with pytest.raises(StressError, match="clean"):
        analyze(_results().query("severity > 0"))


PH_TEXT = "Please cancel order {{Order Number}} and email {{Email Address}} about it because your payment information is wrong."
PH_SPANS = ["{{Order Number}}", "{{Email Address}}"]


@pytest.mark.parametrize("noise_type", list(NOISE_FUNCS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_placeholders_survive_every_noise_type_and_severity_byte_for_byte(noise_type: str, severity: int) -> None:
    from phase2.stress import placeholders

    for i in range(25):  # many seeds so casing/typo/whitespace draws land everywhere
        out = apply_noise(PH_TEXT, f"src-{i}", noise_type, severity)
        found = placeholders(out)
        assert all(s in found for s in PH_SPANS), (noise_type, severity, out)
        assert set(found) == set(PH_SPANS), (noise_type, severity, out)  # nothing partial or new appears


def test_placeholder_guard_is_transparent_without_placeholders_and_rejects_reserved_characters() -> None:
    from phase2.stress import protected

    assert protected(str.upper)("no braces here") == "NO BRACES HERE"
    assert protected(str.upper)("cancel {{Order Number}} now") == "CANCEL {{Order Number}} NOW"
    with pytest.raises(StressError, match="private-use"):
        protected(str.upper)("bad  {{X}}")


def test_frozen_noise_dataset_keeps_every_placeholder_unchanged() -> None:
    from phase2.stress import placeholders

    rows, _ = build()
    with_ph = [r for r in rows if placeholders(r["original_text"])]
    assert len(with_ph) > 300
    for r in with_ph:
        assert all(p in r["text"] for p in placeholders(r["original_text"])), r["variant_id"]
        assert set(placeholders(r["text"])) == set(placeholders(r["original_text"])), r["variant_id"]


def test_noise_reports_primary_valid_only_plus_all_rows_sensitivity_and_questionable_counts() -> None:
    rows = []
    for src, integ in (("s1", "valid"), ("s2", "questionable")):
        rows.append({"provider": "jev", "source_example_id": src, "noise_type": "none", "severity": 0, "semantic_integrity": "valid", "expected_label": "a", "intent": "a", "prediction": "a", "confidence": 0.9, "correct": True, "latency_ms": 1.0})
        rows.append({"provider": "jev", "source_example_id": src, "noise_type": "typos", "severity": 3, "semantic_integrity": integ, "expected_label": "a", "intent": "a", "prediction": "b" if integ == "questionable" else "a", "confidence": 0.5, "correct": integ == "valid", "latency_ms": 1.0})
    tables, summary = analyze(pd.DataFrame(rows))
    valid = tables["noise_valid_only_by_severity"].set_index("severity")
    allrows = tables["noise_all_rows_by_severity"].set_index("severity")
    assert valid.loc[3, "n_pairs"] == 1 and allrows.loc[3, "n_pairs"] == 2  # questionable rows are kept in the sensitivity view
    counts = tables["noise_questionable_counts"].set_index("severity")
    assert counts.loc[3, "valid"] == 1 and counts.loc[3, "questionable"] == 1 and counts.loc[3, "questionable_share"] == 0.5
    assert summary["n_pairs_semantic_integrity_questionable"] == 1 and "all rows" in summary["sensitivity_summary"]
