"""Compute confidence calibration tables from the frozen canonical baseline (no API calls).

Usage:
    uv run python src/phase2/run_calibration.py [--baseline-dir DIR] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.calibration import CALIBRATION_DIR, run_calibration  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 2: confidence calibration.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=CALIBRATION_DIR)
    args = parser.parse_args(argv)
    try:
        meta = run_calibration(args.baseline_dir, args.out)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"eligible rows: {meta['eligible_rows']} of {meta['canonical_rows']} canonical (excluded: {meta['excluded_rows_no_label_or_confidence']})")
    print(f"wrote tables to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
