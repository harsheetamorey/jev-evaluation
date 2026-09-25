"""Synthetic-data tests for multilingual reliability analysis. No network or API access."""

from pathlib import Path

import pandas as pd
import pytest

from dataset_loaders.massive import LOCALES
from phase2.baseline import BaselineError
from phase2.multilingual import (
    CLASSES,
    aligned_records,
    analyze_multilingual,
    classify,
    consistency_summary,
    intent_reliability,
    language_reliability,
    run_multilingual,
)


def make(source: str, pattern: dict[str, bool | None], provider="jev", intent="alarm_set", preds: dict[str, str] | None = None, start_row: int = 0) -> list[dict]:
    rows = []
    for i, loc in enumerate(LOCALES):
        ok = pattern.get(loc)
        rows.append({"raw_row": start_row + i, "run_id": "r1", "experiment": "multilingual", "provider": provider, "example_id": source, "locale": loc, "ground_truth": intent, "candidates": ["a", "b"],
                     "prediction": None if ok is None else (preds or {}).get(loc, intent if ok else "wrong"), "correct": ok, "confidence": None if ok is None else (0.9 if ok else 0.4)})
    return rows


ALL = {loc: True for loc in LOCALES}


def wrong(*locs: str) -> dict[str, bool]:
    return {loc: (loc not in locs) for loc in LOCALES}


def test_classification_is_deterministic_from_correctness_counts() -> None:
    assert classify(8, 0) == "all_correct"
    assert classify(8, 1) == classify(8, 2) == "language_localized_failure"
    assert classify(8, 3) == classify(8, 5) == "mixed_unstable"
    assert classify(8, 6) == classify(8, 8) == "globally_difficult"  # 6/8 = 0.75 is the threshold
    assert classify(5, 0) == "insufficient_locales" and set(CLASSES) >= {"all_correct", "globally_difficult"}


def test_alignment_counts_languages_correct_0_to_8() -> None:
    df = pd.DataFrame(make("s0", wrong(*LOCALES)) + make("s3", wrong("hi-IN", "kn-IN", "ta-IN")) + make("s8", ALL))
    a = aligned_records(df).set_index("source_example_id")
    assert (a.loc["s0", "n_correct"], a.loc["s3", "n_correct"], a.loc["s8", "n_correct"]) == (0, 5, 8)
    assert a.loc["s0", "all_locales_wrong"] and a.loc["s8", "all_locales_correct"] and not a.loc["s3", "all_locales_wrong"]
    assert a.loc["s3", "classification"] == "mixed_unstable" and a.loc["s0", "classification"] == "globally_difficult"
    assert not a.loc["s3", "correct_hi-IN"] and a.loc["s3", "correct_en-US"] and a.loc["s8", "correct_en-US"]


def test_alignment_is_by_source_id_and_locale_across_providers() -> None:
    df = pd.DataFrame(make("s1", ALL, "jev") + make("s1", wrong("ja-JP"), "gpt-4o-mini") + make("s2", ALL, "jev"))
    a = aligned_records(df)
    assert set(zip(a["provider"], a["source_example_id"], strict=True)) == {("jev", "s1"), ("jev", "s2"), ("gpt-4o-mini", "s1")}
    gpt = a[a["provider"] == "gpt-4o-mini"].iloc[0]
    assert gpt["n_correct"] == 7 and gpt["classification"] == "language_localized_failure"


def test_duplicate_locale_rows_are_rejected_explicitly() -> None:
    rows = make("s1", ALL) + make("s1", ALL, start_row=100)
    with pytest.raises(ValueError, match="duplicate"):
        aligned_records(pd.DataFrame(rows))


def test_misaligned_ground_truth_and_unknown_locale_are_rejected() -> None:
    rows = make("s1", ALL)
    rows[0]["ground_truth"] = "other_intent"
    with pytest.raises(ValueError, match="not aligned"):
        aligned_records(pd.DataFrame(rows))
    bad = make("s1", ALL)
    bad[0]["locale"] = "xx-XX"
    with pytest.raises(ValueError, match="unexpected locales"):
        aligned_records(pd.DataFrame(bad))


def test_errored_calls_reduce_available_locales_instead_of_counting_as_wrong() -> None:
    pattern: dict[str, bool | None] = {**ALL, "ja-JP": None, "fr-FR": None, "es-ES": None}  # 3 errored -> 5 available
    a = aligned_records(pd.DataFrame(make("s1", pattern))).iloc[0]
    assert a["n_locales_available"] == 5 and a["n_wrong"] == 0 and a["classification"] == "insufficient_locales" and not a["all_locales_correct"]
    assert a["correct_ja-JP"] is None


def test_prediction_instability_is_flagged_separately_from_correctness() -> None:
    same = aligned_records(pd.DataFrame(make("s1", ALL))).iloc[0]
    diff = aligned_records(pd.DataFrame(make("s2", ALL, preds={"ja-JP": "b"}))).iloc[0]  # 'correct' flag stays as given
    assert not same["prediction_unstable"] and diff["prediction_unstable"] and diff["n_distinct_predictions"] == 2


def test_language_and_intent_tables_show_n_and_hide_small_cells() -> None:
    rows = []
    for i in range(6):
        rows += make(f"s{i}", wrong("ja-JP") if i < 2 else ALL, intent="alarm_set")
    rows += make("t0", ALL, intent="rare_intent")
    df = pd.DataFrame(rows)
    aligned = aligned_records(df)
    lang = language_reliability(df, aligned).set_index("locale")
    assert lang.loc["ja-JP", "n_scored"] == 7 and lang.loc["ja-JP", "accuracy"] == pytest.approx(5 / 7)
    assert lang.loc["ja-JP", "n_wrong_here_only"] == 2 and 0 <= lang.loc["ja-JP", "accuracy_ci95_low"] < lang.loc["ja-JP", "accuracy_ci95_high"] <= 1
    intent, cells = intent_reliability(df, aligned)
    i = intent.set_index("intent")
    assert i.loc["alarm_set", "n_aligned_examples"] == 6 and i.loc["rare_intent", "note"] == "small n: do not rank"
    assert i.loc["alarm_set", "n_examples_language_localized"] == 2
    rare = cells[cells["intent"] == "rare_intent"]
    assert (rare["n"] == 1).all() and (~rare["accuracy_reported"]).all() and rare["accuracy"].isna().all()  # never reported for tiny n
    big = cells[(cells["intent"] == "alarm_set") & (cells["locale"] == "ja-JP")].iloc[0]
    assert big["n"] == 6 and big["accuracy_reported"] and big["accuracy"] == pytest.approx(4 / 6)
    assert not any("rank" in c and "note" != c for c in intent.columns)  # no ranking columns


def test_consistency_summary_counts_and_distribution() -> None:
    df = pd.DataFrame(make("a", ALL) + make("b", ALL) + make("c", wrong(*LOCALES)) + make("d", wrong("ja-JP")))
    s = consistency_summary(aligned_records(df))["providers"]["jev"]
    assert (s["n_aligned_examples"], s["all_8_correct"], s["all_8_wrong"]) == (4, 2, 1)
    assert s["n_correct_distribution"] == {"0": 1, "7": 1, "8": 2}
    assert s["classification_counts"]["language_localized_failure"] == 1 and s["classification_counts"]["globally_difficult"] == 1


def test_analyze_collapses_repeated_measurements_and_reports_population() -> None:
    first = make("s1", wrong("ja-JP"), start_row=0)
    rerun = make("s1", ALL, start_row=100)  # a later re-measurement of the same identities
    canonical = pd.DataFrame(first + rerun)
    tables, summary = analyze_multilingual(canonical)
    a = tables["aligned_reliability"].iloc[0]
    assert a["n_correct"] == 8  # the latest measurement is used, regardless of outcome
    pop = summary["population"]
    assert pop["canonical_multilingual_rows"] == 16 and pop["repeated_measurement_rows_removed"] == 8 and pop["n_aligned_source_ids"] == 1
    assert "not assumed to be 250" in pop["note"] and "not concluded" in summary["interpretation"]


def test_run_multilingual_writes_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    baseline = tmp_path / "b"
    baseline.mkdir()
    pd.DataFrame(make("s1", ALL) + make("s2", wrong("hi-IN"))).to_parquet(baseline / "canonical_results.parquet", index=False)
    out = tmp_path / "out"
    summary = run_multilingual(baseline, out, verify=False)
    assert summary["population"]["n_aligned_source_ids"] == 2 and (out / "aligned_reliability.csv").exists() and (out / "consistency_summary.json").exists()
    with pytest.raises(BaselineError, match="Refusing to overwrite"):
        run_multilingual(baseline, out, verify=False)
