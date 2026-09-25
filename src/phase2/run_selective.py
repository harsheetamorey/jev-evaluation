"""Compute risk-coverage tables from the frozen canonical baseline (no API calls).

Usage:
    uv run python src/phase2/run_selective.py [--baseline-dir DIR] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.selective import SELECTIVE_DIR, run_selective  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 3: selective prediction / risk-coverage.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=SELECTIVE_DIR)
    args = parser.parse_args(argv)
    try:
        summary = run_selective(args.baseline_dir, args.out)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"labeled predictions analyzed: {summary['labeled_rows_analyzed']} of {summary['canonical_rows']} canonical")
    print(f"groups: {summary['n_groups']}; wrote tables to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
