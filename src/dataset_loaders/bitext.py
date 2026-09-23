"""Bitext Customer Support MCQ dataset loader.

Normalizes `crossingminds/bitext_customer_support_mcq` (test split) into the
common {id, text, choices, ground_truth} shape and produces deterministic,
reproducible dev/small/main samples for experiments. Does not call Jev.
"""

import ast
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "crossingminds/bitext_customer_support_mcq"
SPLIT = "test"

SAMPLES_DIR = Path("data/samples")
SAMPLE_SIZES = {"dev": 50, "small": 250, "main": 1000}
DEFAULT_SEED = 42


@dataclass
class BitextRow:
    """One normalized Bitext MCQ example: a message and its five candidate intents."""

    id: str
    text: str
    choices: list[str]
    ground_truth: str


def _parse_choices(raw: str) -> list[str]:
    """`intent_choices` is stored as a stringified Python list literal, not a native list."""
    return ast.literal_eval(raw)


def load_normalized_rows() -> list[BitextRow]:
    """Load and normalize every row of the Bitext MCQ test split (13,436 rows)."""
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    rows = []
    for idx, record in enumerate(dataset):
        choices = _parse_choices(record["intent_choices"])
        rows.append(
            BitextRow(
                id=f"bitext-test-{idx}",
                text=record["instruction"],
                choices=choices,
                ground_truth=choices[record["correct_index"]],
            )
        )
    return rows


def sample_rows(rows: list[BitextRow], n: int, seed: int = DEFAULT_SEED) -> list[BitextRow]:
    """Deterministically sample `n` rows without replacement using `seed`.

    A fresh `Random(seed)` is used per call, so sampling the same `n` from the
    same `rows` always returns the same rows regardless of what else has run.
    """
    rng = random.Random(seed)
    return rng.sample(rows, n)


def save_sample(rows: list[BitextRow], name: str, samples_dir: Path = SAMPLES_DIR) -> Path:
    """Save a normalized sample as JSONL under `samples_dir`."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"bitext_{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    return path


def load_sample(name: str, samples_dir: Path = SAMPLES_DIR) -> list[BitextRow]:
    """Load a previously saved sample from disk."""
    path = samples_dir / f"bitext_{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [BitextRow(**json.loads(line)) for line in f]


def build_all_samples(seed: int = DEFAULT_SEED, samples_dir: Path = SAMPLES_DIR) -> dict[str, Path]:
    """Load the dataset once and write the dev/small/main samples to disk."""
    rows = load_normalized_rows()
    return {
        name: save_sample(sample_rows(rows, size, seed=seed), name, samples_dir=samples_dir)
        for name, size in SAMPLE_SIZES.items()
    }
