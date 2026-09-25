"""Create or verify the Phase II baseline freeze (no API calls).

Usage:
    uv run python src/phase2/freeze_baseline.py create [--results-dir DIR] [--out DIR]
    uv run python src/phase2/freeze_baseline.py verify [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.baseline import (  # noqa: E402
    BASELINE_DIR,
    MANIFEST_NAME,
    PHASE1_RESULTS_DIR,
    BaselineError,
    create_baseline,
    verify_baseline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or verify the Phase I baseline for Phase II.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Copy Phase I outputs and write the manifest (refuses to overwrite).")
    create.add_argument("--results-dir", type=Path, default=PHASE1_RESULTS_DIR)
    create.add_argument("--out", type=Path, default=BASELINE_DIR)
    verify = sub.add_parser("verify", help="Check frozen files and example IDs against the manifest.")
    verify.add_argument("--out", type=Path, default=BASELINE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_baseline(results_dir=args.results_dir, out_dir=args.out)
            unmatched = [r["run_id"] for r in manifest["runs"] if r["config"] is None]
            print(f"baseline written: {args.out / MANIFEST_NAME}")
            print(f"files frozen: {len(manifest['files'])}, runs recorded: {len(manifest['runs'])}")
            canon = manifest["canonical"]
            if canon:
                print(f"raw rows: {canon['raw_rows']}, canonical rows: {canon['canonical_rows']}, removed: {canon['rows_removed']}, disagreeing repeats: {canon['disagreeing_repeated_calls']}/{canon['duplicated_identities']}")
            if unmatched:
                print(f"note: config could not be recovered for run_id(s): {', '.join(unmatched)}")
            return 0
        errors, warnings = verify_baseline(args.out / MANIFEST_NAME)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    print("baseline verified" if not errors else "baseline verification FAILED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
