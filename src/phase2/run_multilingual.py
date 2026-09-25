"""Compute multilingual reliability tables from the frozen canonical baseline (no API calls).

Usage:
    uv run python src/phase2/run_multilingual.py [--baseline-dir DIR] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.multilingual import MULTILINGUAL_DIR, run_multilingual  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 12: multilingual reliability deep dive.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=MULTILINGUAL_DIR)
    args = parser.parse_args(argv)
    try:
        summary = run_multilingual(args.baseline_dir, args.out)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pop = summary["population"]
    print(f"aligned source ids: {pop['n_aligned_source_ids']}, intents: {pop['n_intents']}, locales: {len(pop['locales_found'])}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
