"""MASSIVE multilingual dataset loader.

Builds aligned cross-language samples: the same MASSIVE `id` evaluated across
every selected locale, so predictions can be compared for the same underlying
semantic utterance. Loaded from the dataset's Hugging-Face-hosted Parquet
conversion (`refs/convert/parquet`), since the AmazonScience/massive repo
itself only ships a legacy loading script that current `datasets` versions
reject outright.
"""

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

from models.prediction import Example

DATASET_NAME = "AmazonScience/massive"
DATASET_REVISION = "refs/convert/parquet"
PARTITION = "test"

LOCALES = ["en-US", "hi-IN", "kn-IN", "ta-IN", "te-IN", "es-ES", "fr-FR", "ja-JP"]

SAMPLES_DIR = Path("data/samples")
ALIGNED_SAMPLE_SIZES = {"aligned_100": 100, "aligned_250": 250}
DEFAULT_SEED = 42

# Choice-scaling experiment (K candidate intents): en-US test-partition intent
# frequencies range from 1 to 209 rows (median 35), so an arbitrary or random K=5
# subset risks landing on a near-empty intent. Ranking by frequency and taking
# prefixes keeps every K's pool as large as possible while staying deterministic.
CHOICE_SCALING_LOCALE = "en-US"
CHOICE_SCALING_K_VALUES = [5, 10, 25, 60]
CHOICE_SCALING_SAMPLE_SIZE = 100

# The 60 MASSIVE intent labels, captured from the dataset's own ClassLabel feature
# (order matters: it's how `intent` integer codes map back to names).
INTENTS = [
    "datetime_query", "iot_hue_lightchange", "transport_ticket", "takeaway_query",
    "qa_stock", "general_greet", "recommendation_events", "music_dislikeness",
    "iot_wemo_off", "cooking_recipe", "qa_currency", "transport_traffic",
    "general_quirky", "weather_query", "audio_volume_up", "email_addcontact",
    "takeaway_order", "email_querycontact", "iot_hue_lightup", "recommendation_locations",
    "play_audiobook", "lists_createoradd", "news_query", "alarm_query",
    "iot_wemo_on", "general_joke", "qa_definition", "social_query",
    "music_settings", "audio_volume_other", "calendar_remove", "iot_hue_lightdim",
    "calendar_query", "email_sendemail", "iot_cleaning", "audio_volume_down",
    "play_radio", "cooking_query", "datetime_convert", "qa_maths",
    "iot_hue_lightoff", "iot_hue_lighton", "transport_query", "music_likeness",
    "email_query", "play_music", "audio_volume_mute", "social_post",
    "alarm_set", "qa_factoid", "calendar_set", "play_game",
    "alarm_remove", "lists_remove", "transport_taxi", "recommendation_movies",
    "iot_coffee", "music_query", "play_podcasts", "lists_query",
]  # fmt: skip


@dataclass
class MassiveRow:
    """One normalized MASSIVE example: an utterance in one locale and its intent."""

    id: str
    locale: str
    text: str
    ground_truth: str


def _load_locale_partition(locale: str, partition: str = PARTITION) -> Dataset:
    """Download and load one locale's partition from the dataset's Parquet conversion."""
    path = hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=f"{locale}/{partition}/0000.parquet",
    )
    return Dataset.from_parquet(path)


def _normalize_row(record: dict, locale: str, intent_names: list[str]) -> MassiveRow:
    """Normalize one raw dataset record, refusing anything outside the test partition."""
    if record["partition"] != PARTITION:
        raise ValueError(
            f"{locale}: row id={record['id']!r} is from partition {record['partition']!r}, "
            f"not {PARTITION!r} -- refusing to let it leak into evaluation."
        )
    return MassiveRow(
        id=record["id"],
        locale=locale,
        text=record["utt"],
        ground_truth=intent_names[record["intent"]],
    )


def load_normalized_rows(locale: str) -> list[MassiveRow]:
    """Load and normalize every test-partition row for one locale."""
    dataset = _load_locale_partition(locale)
    intent_names = dataset.features["intent"].names
    if intent_names != INTENTS:
        raise ValueError(f"{locale}: dataset's intent label set does not match the expected 60 MASSIVE intents.")
    return [_normalize_row(record, locale, intent_names) for record in dataset]


def load_all_locales(locales: list[str] = LOCALES) -> dict[str, list[MassiveRow]]:
    """Load and normalize the test partition for every locale in `locales`."""
    return {locale: load_normalized_rows(locale) for locale in locales}


def shared_ids(rows_by_locale: dict[str, list[MassiveRow]]) -> set[str]:
    """IDs present in every locale's rows -- the only ones usable for aligned sampling."""
    id_sets = [{row.id for row in rows} for rows in rows_by_locale.values()]
    return set.intersection(*id_sets) if id_sets else set()


def sample_aligned_ids(ids: set[str] | list[str], n: int, seed: int = DEFAULT_SEED) -> list[str]:
    """Deterministically sample `n` shared IDs using `seed`."""
    rng = random.Random(seed)
    return rng.sample(sorted(ids), n)


def validate_aligned_consistency(rows: list[MassiveRow]) -> None:
    """Raise if any aligned ID doesn't share the same ground_truth intent across locales."""
    truth_by_id: dict[str, str] = {}
    for row in rows:
        expected = truth_by_id.setdefault(row.id, row.ground_truth)
        if row.ground_truth != expected:
            raise ValueError(
                f"id={row.id!r}: ground_truth mismatch across locales ({row.locale}={row.ground_truth!r} vs {expected!r})."
            )


def build_aligned_sample(rows_by_locale: dict[str, list[MassiveRow]], ids: list[str]) -> list[MassiveRow]:
    """Build the aligned sample: every selected ID's row, for every locale, grouped by ID.

    Raises if a selected ID is missing for any locale, or if locales disagree on an
    aligned ID's ground-truth intent.
    """
    by_locale_by_id = {locale: {row.id: row for row in rows} for locale, rows in rows_by_locale.items()}
    for locale, by_id in by_locale_by_id.items():
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"{locale}: missing {len(missing)} of the selected IDs, e.g. {missing[:5]}.")

    rows = [by_locale_by_id[locale][i] for i in ids for locale in rows_by_locale]
    validate_aligned_consistency(rows)
    return rows


def to_example(row: MassiveRow, candidates: list[str] | None = None) -> Example:
    """Adapt a MassiveRow into the generic Example shape evaluators consume.

    Defaults to classifying among all 60 MASSIVE intents; pass a narrower
    `candidates` list for experiments that vary the candidate-set size (see
    the choice-scaling functions below). Either way, `row.ground_truth` must
    be a member of `candidates` for the example to be meaningful.
    """
    return Example(
        example_id=row.id,
        dataset="massive",
        state=row.text,
        candidates=candidates if candidates is not None else INTENTS,
        ground_truth=row.ground_truth,
        locale=row.locale,
    )


def rank_intents_by_frequency(rows: list[MassiveRow]) -> list[str]:
    """Order all 60 intents by descending frequency in `rows`, ties broken alphabetically.

    Deterministic and independent of `rows`' order -- only the counts matter.
    """
    counts = Counter(row.ground_truth for row in rows)
    return sorted(INTENTS, key=lambda intent: (-counts.get(intent, 0), intent))


def build_choice_scaling_subsets(
    intent_order: list[str],
    k_values: list[int] = CHOICE_SCALING_K_VALUES,
) -> dict[int, list[str]]:
    """K-sized prefixes of `intent_order`, so smaller-K subsets nest inside larger ones."""
    return {k: intent_order[:k] for k in k_values}


def select_choice_scaling_sample(
    rows: list[MassiveRow],
    smallest_subset: list[str],
    n: int = CHOICE_SCALING_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[MassiveRow]:
    """Deterministically sample `n` rows whose ground_truth is in `smallest_subset` (K=5).

    Because every larger K's subset is a superset of the smallest one, this exact
    sample of examples stays valid -- and identical -- at every K, isolating
    candidate-set size as the only variable across the comparison.
    """
    pool = [row for row in rows if row.ground_truth in smallest_subset]
    if len(pool) < n:
        raise ValueError(
            f"Only {len(pool)} examples have ground_truth in the smallest intent subset "
            f"{smallest_subset!r}; need at least {n}."
        )
    return random.Random(seed).sample(pool, n)


def save_sample(rows: list[MassiveRow], name: str, samples_dir: Path = SAMPLES_DIR) -> Path:
    """Save a normalized aligned sample as JSONL under `samples_dir`."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"massive_{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    return path


def load_sample(name: str, samples_dir: Path = SAMPLES_DIR) -> list[MassiveRow]:
    """Load a previously saved aligned sample from disk."""
    path = samples_dir / f"massive_{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [MassiveRow(**json.loads(line)) for line in f]


def build_all_samples(
    locales: list[str] = LOCALES,
    seed: int = DEFAULT_SEED,
    samples_dir: Path = SAMPLES_DIR,
) -> dict[str, Path]:
    """Load every locale once and write the aligned_100/aligned_250 samples to disk."""
    rows_by_locale = load_all_locales(locales)
    common_ids = shared_ids(rows_by_locale)
    paths = {}
    for name, size in ALIGNED_SAMPLE_SIZES.items():
        ids = sample_aligned_ids(common_ids, size, seed=seed)
        sample = build_aligned_sample(rows_by_locale, ids)
        paths[name] = save_sample(sample, name, samples_dir=samples_dir)
    return paths
