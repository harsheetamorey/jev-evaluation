"""Build the choice-scaling base sample and frequency-ranked intent order.

Saves data/samples/massive_choice_scaling.jsonl (the 100 examples, ground_truth
restricted to MASSIVE's 5 most frequent en-US intents) and
data/samples/massive_choice_scaling_intent_order.json (the full 60-intent
frequency ranking and its K=5/10/25/60 prefixes).

Usage:
    uv run python scripts/build_choice_scaling_sample.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_loaders.massive import (  # noqa: E402
    CHOICE_SCALING_K_VALUES,
    CHOICE_SCALING_LOCALE,
    CHOICE_SCALING_SAMPLE_SIZE,
    DEFAULT_SEED,
    SAMPLES_DIR,
    build_choice_scaling_subsets,
    load_normalized_rows,
    rank_intents_by_frequency,
    save_sample,
    select_choice_scaling_sample,
)


def main() -> None:
    rows = load_normalized_rows(CHOICE_SCALING_LOCALE)
    intent_order = rank_intents_by_frequency(rows)
    subsets = build_choice_scaling_subsets(intent_order, CHOICE_SCALING_K_VALUES)

    smallest_k = min(CHOICE_SCALING_K_VALUES)
    sample = select_choice_scaling_sample(rows, subsets[smallest_k], n=CHOICE_SCALING_SAMPLE_SIZE, seed=DEFAULT_SEED)
    sample_path = save_sample(sample, "choice_scaling", samples_dir=SAMPLES_DIR)

    order_path = SAMPLES_DIR / "massive_choice_scaling_intent_order.json"
    order_path.write_text(
        json.dumps({"locale": CHOICE_SCALING_LOCALE, "intent_order": intent_order, "subsets": subsets}, indent=2)
    )

    print(f"sample: {sample_path} ({len(sample)} rows)")
    print(f"intent order: {order_path}")
    for k in CHOICE_SCALING_K_VALUES:
        print(f"  K={k}: {subsets[k]}")


if __name__ == "__main__":
    main()
