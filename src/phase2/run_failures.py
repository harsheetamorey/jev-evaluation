"""Build the normalized Phase II failure dataset from recorded results (no model calls).

Usage:
    uv run python src/phase2/run_failures.py [--baseline-dir DIR] [--out DIR] [--high 0.90] [--low 0.60]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.failures import FAILURES_DIR, HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD, run_failures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 13: failure taxonomy / Failure Museum data.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=FAILURES_DIR)
    parser.add_argument("--high", type=float, default=HIGH_CONFIDENCE_THRESHOLD, help="high_confidence_wrong threshold (configurable slicing choice)")
    parser.add_argument("--low", type=float, default=LOW_CONFIDENCE_THRESHOLD, help="low_confidence_wrong threshold (configurable slicing choice)")
    args = parser.parse_args(argv)
    try:
        s = run_failures(args.baseline_dir, out_dir=args.out, high=args.high, low=args.low)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"failures: {s['total_failures']}; unclassified: {s['n_unclassified_no_tag']}; experiments without results: {', '.join(s['experiments_with_no_recorded_results']) or 'none'}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
