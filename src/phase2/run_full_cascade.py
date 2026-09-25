"""Offline Rules -> Jev -> LLM -> oracle cascade simulation (no API calls; the 'human' is an oracle).

Usage:
    uv run python src/phase2/run_full_cascade.py [--jev-threshold 0.9] [--llm-threshold 0.5] [--no-llm] [--no-oracle]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import BASELINE_DIR, BaselineError  # noqa: E402
from phase2.full_cascade import FULL_CASCADE_DIR, CascadeConfig, run_full_cascade  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II Step 15: full decision cascade simulation.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out", type=Path, default=FULL_CASCADE_DIR)
    parser.add_argument("--jev-threshold", type=float, default=0.90, help="configuration, not a recommendation")
    parser.add_argument("--llm-threshold", type=float, default=0.5, help="LLM output accepted iff confidence >= this (configuration, not a recommendation)")
    parser.add_argument("--no-llm", action="store_true", help="disable the LLM stage")
    parser.add_argument("--no-oracle", action="store_true", help="disable the oracle stage (unresolved requests stay unhandled)")
    args = parser.parse_args(argv)
    cfg = CascadeConfig(jev_threshold=args.jev_threshold, llm_enabled=not args.no_llm, llm_threshold=args.llm_threshold, oracle_enabled=not args.no_oracle)
    try:
        s = run_full_cascade(args.baseline_dir, args.out, cfg)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"requests: {s['n_requests']}; rules coverage: {s['rules']['actual_coverage']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
