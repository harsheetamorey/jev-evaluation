"""Tests for the Bitext MCQ dataset loader.

Loads the real dataset from Hugging Face (public, no auth) once per test run
and caches locally afterward via the `datasets` library's own cache.
"""

from pathlib import Path

import pytest

from dataset_loaders.bitext import (
    SAMPLE_SIZES,
    BitextRow,
    load_normalized_rows,
    load_sample,
    sample_rows,
    save_sample,
)


@pytest.fixture(scope="module")
def rows() -> list[BitextRow]:
    return load_normalized_rows()


def test_every_row_has_five_choices(rows: list[BitextRow]) -> None:
    assert all(len(row.choices) == 5 for row in rows)


def test_ground_truth_belongs_to_choices(rows: list[BitextRow]) -> None:
    assert all(row.ground_truth in row.choices for row in rows)


def test_ids_are_unique(rows: list[BitextRow]) -> None:
    ids = [row.id for row in rows]
    assert len(ids) == len(set(ids))


def test_sample_generation_is_deterministic(rows: list[BitextRow]) -> None:
    first = sample_rows(rows, 50, seed=42)
    second = sample_rows(rows, 50, seed=42)
    assert [row.id for row in first] == [row.id for row in second]


def test_sample_generation_differs_by_seed(rows: list[BitextRow]) -> None:
    seed_42 = sample_rows(rows, 50, seed=42)
    seed_7 = sample_rows(rows, 50, seed=7)
    assert [row.id for row in seed_42] != [row.id for row in seed_7]


@pytest.mark.parametrize("size", list(SAMPLE_SIZES.values()))
def test_sample_size_matches_requested(rows: list[BitextRow], size: int) -> None:
    sample = sample_rows(rows, size, seed=42)
    assert len(sample) == size


def test_save_and_load_sample_round_trips(tmp_path: Path, rows: list[BitextRow]) -> None:
    sample = sample_rows(rows, 10, seed=42)

    save_sample(sample, "unit-test", samples_dir=tmp_path)
    loaded = load_sample("unit-test", samples_dir=tmp_path)

    assert loaded == sample
