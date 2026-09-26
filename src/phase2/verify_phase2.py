"""Phase II reproducibility / integrity CLI (no model calls).

    uv run python src/phase2/verify_phase2.py verify [--skip-regeneration]   # integrity + regeneration + stress datasets + offline pipelines
    uv run python src/phase2/verify_phase2.py plan [--markdown docs/phase2-live-reproduction.md]   # live-call counts + commands (nothing is run)
    uv run python src/phase2/verify_phase2.py manifest                          # write data/results/phase2/phase2_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.integrity import (  # noqa: E402
    DOCS_PLAN,
    MANIFEST_OUT,
    live_reproduction_plan,
    pipeline_selfcheck,
    regeneration_checks,
    render_plan_markdown,
    repository_checks,
    stress_dataset_checks,
    write_phase2_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase II reproducibility and integrity pass.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="run every integrity and reproducibility check (no API calls)")
    verify.add_argument("--skip-regeneration", action="store_true")
    plan = sub.add_parser("plan", help="live-call counts, cost estimates and commands (nothing is executed)")
    plan.add_argument("--markdown", type=Path, nargs="?", const=DOCS_PLAN, default=None, help="also write the plan as markdown (refuses to overwrite)")
    sub.add_parser("manifest", help=f"write {MANIFEST_OUT} (refuses to overwrite)")
    args = parser.parse_args(argv)

    if args.command == "verify":
        results = repository_checks() + stress_dataset_checks() + pipeline_selfcheck() + ([] if args.skip_regeneration else regeneration_checks())
        for r in results:
            print(f"[{r['status'].upper():4}] {r['check']}" + (f"\n         {r['detail']}" if r["detail"] else ""))
        counts = {s: sum(r["status"] == s for r in results) for s in ("pass", "fail", "warn", "info")}
        print(f"\n{counts}")
        return 1 if counts["fail"] else 0
    if args.command == "plan":
        plan_data = live_reproduction_plan()
        print(json.dumps({k: v for k, v in plan_data.items() if k != "datasets"}, indent=2))
        if args.markdown:
            if args.markdown.exists():
                print(f"error: {args.markdown} exists; refusing to overwrite", file=sys.stderr)
                return 1
            args.markdown.write_text(render_plan_markdown(plan_data))
            print(f"wrote {args.markdown}")
        return 0
    try:
        m = write_phase2_manifest()
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {MANIFEST_OUT} (git {m['git']['commit'][:10] if m['git']['commit'] else None})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
