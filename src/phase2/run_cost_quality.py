"""Cost x quality comparison across architectures from the frozen canonical baseline (no API calls).

Usage:
    uv run python src/phase2/run_cost_quality.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.cost_quality import COST_QUALITY_DIR, run_cost_quality  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 16: cost x quality frontier.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=COST_QUALITY_DIR)
    args = parser.parse_args(argv)
    try:
        s = run_cost_quality(args.baseline_dir, args.out)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"workloads: {len(s['workloads'])}, points: {s['n_points']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
