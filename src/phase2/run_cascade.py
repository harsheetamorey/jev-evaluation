"""Offline Jev -> LLM cascade simulation from the frozen canonical baseline (no API calls).

Usage:
    uv run python src/phase2/run_cascade.py [--thresholds 0.6,0.7,0.8,0.9,0.95] [--sweep] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.cascade import CASCADE_DIR, DEFAULT_THRESHOLDS, run_cascade  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 14: Jev -> LLM cascade simulation.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=CASCADE_DIR)
    parser.add_argument("--thresholds", type=lambda s: [float(x) for x in s.split(",")], default=DEFAULT_THRESHOLDS)
    parser.add_argument("--sweep", action="store_true", help="also compute summary rows for every 0.01 threshold from 0.50 to 0.99")
    args = parser.parse_args(argv)
    try:
        s = run_cascade(args.baseline_dir, args.out, args.thresholds, args.sweep)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"matched examples: {s['n_matched_examples']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
