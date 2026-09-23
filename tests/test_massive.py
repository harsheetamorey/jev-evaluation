"""Tests for the MASSIVE multilingual dataset loader.

Loads the real dataset (all 8 target locales, test partition) from Hugging
Face once per test run, cached locally afterward via `huggingface_hub`.
"""

from pathlib import Path

import pytest

from dataset_loaders.massive import (
    INTENTS,
    LOCALES,
    MassiveRow,
    _normalize_row,
    build_aligned_sample,
    build_choice_scaling_subsets,
    load_all_locales,
    load_sample,
    rank_intents_by_frequency,
    sample_aligned_ids,
    save_sample,
    select_choice_scaling_sample,
    shared_ids,
    to_example,
    validate_aligned_consistency,
)


@pytest.fixture(scope="module")
def rows_by_locale() -> dict[str, list[MassiveRow]]:
    return load_all_locales()


def test_every_target_locale_is_loaded(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    assert set(rows_by_locale) == set(LOCALES)
    assert all(len(rows) > 0 for rows in rows_by_locale.values())


def test_every_ground_truth_is_one_of_60_intents(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    assert len(INTENTS) == 60
    for rows in rows_by_locale.values():
        assert all(row.ground_truth in INTENTS for row in rows)


def test_shared_ids_cover_the_full_aligned_test_set(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    common = shared_ids(rows_by_locale)
    any_locale_size = len(next(iter(rows_by_locale.values())))
    assert len(common) == any_locale_size


def test_sample_generation_is_deterministic(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    common = shared_ids(rows_by_locale)
    first = sample_aligned_ids(common, 100, seed=42)
    second = sample_aligned_ids(common, 100, seed=42)
    assert first == second


def test_sample_generation_differs_by_seed(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    common = shared_ids(rows_by_locale)
    seed_42 = sample_aligned_ids(common, 100, seed=42)
    seed_7 = sample_aligned_ids(common, 100, seed=7)
    assert seed_42 != seed_7


def test_aligned_sample_has_every_locale_for_every_selected_id(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    ids = sample_aligned_ids(shared_ids(rows_by_locale), 20, seed=42)
    sample = build_aligned_sample(rows_by_locale, ids)

    assert len(sample) == 20 * len(LOCALES)
    locales_by_id: dict[str, set[str]] = {}
    for row in sample:
        locales_by_id.setdefault(row.id, set()).add(row.locale)
    assert set(locales_by_id) == set(ids)
    assert all(locales_seen == set(LOCALES) for locales_seen in locales_by_id.values())


def test_aligned_sample_id_locale_pairs_are_unique(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    ids = sample_aligned_ids(shared_ids(rows_by_locale), 20, seed=42)
    sample = build_aligned_sample(rows_by_locale, ids)
    pairs = [(row.id, row.locale) for row in sample]
    assert len(pairs) == len(set(pairs))


def test_build_aligned_sample_raises_if_id_missing_for_a_locale(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    truncated = dict(rows_by_locale)
    truncated["ja-JP"] = [row for row in rows_by_locale["ja-JP"] if row.id != "0"]
    with pytest.raises(ValueError):
        build_aligned_sample(truncated, ["0"])


def test_real_aligned_sample_passes_consistency_validation(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    ids = sample_aligned_ids(shared_ids(rows_by_locale), 20, seed=42)
    sample = build_aligned_sample(rows_by_locale, ids)  # raises internally if inconsistent
    validate_aligned_consistency(sample)  # should not raise


def test_validate_aligned_consistency_raises_on_mismatched_ground_truth() -> None:
    rows = [
        MassiveRow(id="1", locale="en-US", text="a", ground_truth="alarm_set"),
        MassiveRow(id="1", locale="hi-IN", text="b", ground_truth="alarm_query"),
    ]
    with pytest.raises(ValueError):
        validate_aligned_consistency(rows)


def test_normalize_row_rejects_non_test_partition() -> None:
    record = {"id": "5", "utt": "hello", "intent": 0, "partition": "train"}
    with pytest.raises(ValueError):
        _normalize_row(record, "en-US", INTENTS)


def test_normalize_row_maps_fields_correctly() -> None:
    record = {"id": "5", "utt": "hello", "intent": 3, "partition": "test"}
    row = _normalize_row(record, "en-US", INTENTS)
    assert row == MassiveRow(id="5", locale="en-US", text="hello", ground_truth=INTENTS[3])


def test_to_example_uses_all_60_intents_as_candidates(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    row = rows_by_locale["hi-IN"][0]
    example = to_example(row)
    assert example.example_id == row.id
    assert example.locale == "hi-IN"
    assert example.candidates == INTENTS
    assert example.ground_truth == row.ground_truth


def test_to_example_accepts_a_narrower_candidate_list(rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    row = rows_by_locale["en-US"][0]
    subset = [row.ground_truth, "alarm_set", "play_music"]
    example = to_example(row, candidates=subset)
    assert example.candidates == subset


def test_rank_intents_by_frequency_is_deterministic_and_covers_all_60() -> None:
    rows = [
        MassiveRow(id="1", locale="en-US", text="a", ground_truth="alarm_set"),
        MassiveRow(id="2", locale="en-US", text="b", ground_truth="alarm_set"),
        MassiveRow(id="3", locale="en-US", text="c", ground_truth="play_music"),
    ]
    order_a = rank_intents_by_frequency(rows)
    order_b = rank_intents_by_frequency(rows)
    assert order_a == order_b
    assert set(order_a) == set(INTENTS)
    assert order_a[0] == "alarm_set"  # 2 occurrences, most frequent
    assert order_a[1] == "play_music"  # 1 occurrence, next most frequent


def test_rank_intents_by_frequency_breaks_ties_alphabetically() -> None:
    # No rows at all -> every intent is a 0-count tie -> alphabetical order.
    order = rank_intents_by_frequency([])
    assert order == sorted(INTENTS)


def test_build_choice_scaling_subsets_are_nested_prefixes() -> None:
    order = sorted(INTENTS)  # any fixed order works for this structural check
    subsets = build_choice_scaling_subsets(order, k_values=[5, 10, 25, 60])

    assert [len(subsets[k]) for k in (5, 10, 25, 60)] == [5, 10, 25, 60]
    assert set(subsets[5]) <= set(subsets[10]) <= set(subsets[25]) <= set(subsets[60])
    assert subsets[60] == order


def test_select_choice_scaling_sample_is_deterministic_and_within_subset() -> None:
    rows = [MassiveRow(id=str(i), locale="en-US", text=str(i), ground_truth="alarm_set") for i in range(20)]
    rows += [MassiveRow(id="other", locale="en-US", text="x", ground_truth="play_music")]

    first = select_choice_scaling_sample(rows, ["alarm_set"], n=10, seed=42)
    second = select_choice_scaling_sample(rows, ["alarm_set"], n=10, seed=42)

    assert [row.id for row in first] == [row.id for row in second]
    assert all(row.ground_truth == "alarm_set" for row in first)


def test_select_choice_scaling_sample_raises_if_pool_too_small() -> None:
    rows = [MassiveRow(id="1", locale="en-US", text="a", ground_truth="alarm_set")]
    with pytest.raises(ValueError, match="Only 1 examples"):
        select_choice_scaling_sample(rows, ["alarm_set"], n=10, seed=42)


def test_save_and_load_sample_round_trips(tmp_path: Path, rows_by_locale: dict[str, list[MassiveRow]]) -> None:
    ids = sample_aligned_ids(shared_ids(rows_by_locale), 5, seed=42)
    sample = build_aligned_sample(rows_by_locale, ids)

    save_sample(sample, "unit-test", samples_dir=tmp_path)
    loaded = load_sample("unit-test", samples_dir=tmp_path)

    assert loaded == sample
