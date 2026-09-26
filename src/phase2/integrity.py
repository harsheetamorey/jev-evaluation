"""Phase II Step 18: reproducibility and integrity checks (adds no experiments, makes no model calls).

Proves that Phase I was preserved and that Phase II is reproducible from frozen inputs:

  * Phase I result files still match the hashes recorded in the frozen baseline manifest
  * the frozen baseline verifies and the canonical population is what the manifest says
  * tracked manifests/datasets match what is on disk; no secrets or .env files are tracked
  * every analysis-only Phase II artifact regenerates byte-for-byte from the frozen inputs
  * every frozen stress dataset verifies (hash, unique ids, source_example_id, stored seeds/config, deterministic rebuild)
  * the whole build -> run -> analyze pipeline works for every stress experiment against a FAKE offline client
  * the exact live-call counts, cost estimates and commands needed to reproduce the live experiments
  * a Phase II manifest that records what was and was NOT recorded historically (never back-filled)
"""

import asyncio
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, NOT_RECORDED, load_manifest, sha256_file, verify_baseline
from phase2.stress import PHASE2_RESULTS_DIR, STRESS_DIR, ExperimentSpec, execute_live, load_frozen_dataset, read_results

PHASE1_DIR = Path("data/results")
MANIFEST_OUT = PHASE2_RESULTS_DIR / "phase2_manifest.json"
DOCS_PLAN = Path("docs/phase2-live-reproduction.md")
SECRET_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9]{20,}", r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{24,}")]
STRESS_MODULES = {"ambiguity": "phase2.ambiguity", "stability": "phase2.stability", "noise": "phase2.noise", "context_pollution": "phase2.context_pollution", "context_relevance": "phase2.context_relevance", "ood": "phase2.ood", "adversarial": "phase2.adversarial", "choice_overlap": "phase2.choice_overlap"}
STRESS_CLI = {"ambiguity": "run_ambiguity.py", "stability": "run_stability.py", "noise": "run_noise.py", "context_pollution": "run_context_pollution.py", "context_relevance": "run_context_relevance.py", "ood": "run_ood.py", "adversarial": "run_adversarial.py", "choice_overlap": "run_choice_overlap.py"}
SEED_KEYS = ("seed", "context_seed")
ANALYSIS_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "calibration": ("calibration_bins.csv", "calibration_summary.csv", "calibration_meta.json"),
    "selective_prediction": ("risk_coverage.csv", "risk_coverage_summary.json"),
    "multilingual": ("aligned_reliability.csv", "intent_reliability.csv", "intent_language_reliability.csv", "language_reliability.csv", "consistency_summary.json"),
    "failures": ("failures.jsonl", "failures.csv", "failures_summary.json"),
    "cascade": ("coverage_audit.csv", "missing_llm_report.json", "cascade_thresholds.csv", "cascade_per_example.csv", "cascade_summary.json"),
    "full_cascade": ("full_cascade_per_request.csv", "full_cascade_grid.csv", "full_cascade_summary.json"),
    "cost_quality": ("architecture_frontier.csv", "cost_accuracy_points.csv", "frontier_summary.json"),
}


def check(name: str, status: str, detail: str = "") -> dict[str, str]:
    assert status in {"pass", "fail", "warn", "info"}
    return {"check": name, "status": status, "detail": detail}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- 1. repository integrity ---------------------------------------------------------------------


def repository_checks(baseline_dir: Path = BASELINE_DIR) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    manifest_path = baseline_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    changed = [n for n, i in manifest["files"].items() if not Path(i["source"]).exists() or sha256_file(Path(i["source"])) != i["sha256"]]
    out.append(check("Phase I result files unchanged since the freeze", "fail" if changed else "pass", f"changed/missing: {changed}" if changed else f"{len(manifest['files'])} files match their recorded sha256"))
    errors, warnings = verify_baseline(manifest_path)
    out.append(check("frozen baseline verifies (files, derived files, example IDs, canonical re-derivation)", "fail" if errors else "pass", "; ".join(errors) or "ok"))
    if warnings:
        out.append(check("baseline warnings (Phase I sources or samples differ from the freeze)", "warn", "; ".join(warnings)))
    canonical = pd.read_parquet(baseline_dir / CANONICAL_NAME)
    c = manifest["canonical"]
    ok = len(canonical) == c["canonical_rows"] and c["raw_rows"] - c["canonical_rows"] == c["rows_removed"]
    out.append(check("canonical baseline holds the expected population", "pass" if ok else "fail", f"{len(canonical)} rows (manifest: {c['canonical_rows']}; raw {c['raw_rows']}; removed {c['rows_removed']})"))
    per_run_bad = [r["run_id"] for r in manifest["runs"] if int((canonical["run_id"] == r["run_id"]).sum()) != r["canonical_rows"]]
    out.append(check("per-run canonical row counts match the manifest", "fail" if per_run_bad else "pass", f"mismatch: {per_run_bad}" if per_run_bad else f"{len(manifest['runs'])} runs"))
    tracked = [str(baseline_dir / MANIFEST_NAME), str(baseline_dir / "canonical_dedup_report.csv"), str(STRESS_DIR)]
    dirty = _git("status", "--porcelain", "--", *tracked)
    out.append(check("tracked manifests/reports/datasets match what is on disk (git)", "info" if dirty is None else ("fail" if dirty else "pass"), "git unavailable" if dirty is None else (dirty or "no uncommitted difference")))
    files = (_git("ls-files") or "").splitlines()
    env_tracked = [f for f in files if re.search(r"(^|/)\.env($|\.(?!example$))", f)]
    out.append(check("no .env file is tracked", "fail" if env_tracked else "pass", str(env_tracked) if env_tracked else ".env.example only"))
    hits = []
    for f in files:
        p = Path(f)
        if p.suffix in {".py", ".md", ".toml", ".json", ".jsonl", ".csv", ".txt", ".example", ".yml", ".yaml"} and p.exists() and p.stat().st_size < 3_000_000 and "tests/" not in f:
            text = p.read_text(errors="ignore")
            hits += [f"{f}: {m.group(0)[:12]}..." for pat in SECRET_PATTERNS for m in pat.finditer(text)]
    out.append(check("no API-key-like strings in tracked source, data or docs", "fail" if hits else "pass", "; ".join(hits[:5]) if hits else f"{len(files)} tracked files scanned (tests excluded: they use blank keys)"))
    return out


# --- 2. deterministic regeneration ---------------------------------------------------------------


def _diff_detail(a: Path, b: Path) -> str:
    la, lb = a.read_bytes().splitlines(), b.read_bytes().splitlines()
    for i, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x != y:
            return f"first difference at line {i + 1}"
    return f"line counts differ ({len(la)} vs {len(lb)})"


def regeneration_checks(baseline_dir: Path = BASELINE_DIR, existing_root: Path = PHASE2_RESULTS_DIR) -> list[dict[str, str]]:
    from phase2.calibration import run_calibration
    from phase2.cascade import run_cascade
    from phase2.cost_quality import run_cost_quality
    from phase2.failures import run_failures
    from phase2.full_cascade import run_full_cascade
    from phase2.multilingual import run_multilingual
    from phase2.selective import run_selective

    out: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        runners = {
            "calibration": lambda d: run_calibration(baseline_dir, d, verify=False),
            "selective_prediction": lambda d: run_selective(baseline_dir, d, verify=False),
            "multilingual": lambda d: run_multilingual(baseline_dir, d, verify=False),
            "failures": lambda d: run_failures(baseline_dir, existing_root, d, verify=False),
            "cascade": lambda d: run_cascade(baseline_dir, d, verify=False),
            "full_cascade": lambda d: run_full_cascade(baseline_dir, d, verify=False),
            "cost_quality": lambda d: run_cost_quality(baseline_dir, d, verify=False),
        }
        for name, run in runners.items():
            existing_dir = existing_root / name
            if not existing_dir.exists():
                out.append(check(f"regenerate {name}", "warn", f"no existing output in {existing_dir}; nothing to compare"))
                continue
            regen_dir = tmp_root / name
            run(regen_dir)
            diffs = []
            for fname in ANALYSIS_ARTIFACTS[name]:
                old, new = existing_dir / fname, regen_dir / fname
                if not old.exists():
                    diffs.append(f"{fname}: missing from existing output")
                elif old.read_bytes() != new.read_bytes():
                    diffs.append(f"{fname}: {_diff_detail(old, new)}")
            out.append(check(f"regenerate {name} and compare byte-for-byte", "fail" if diffs else "pass", "; ".join(diffs) if diffs else f"{len(ANALYSIS_ARTIFACTS[name])} files identical"))
    return out


# --- 3. stress datasets --------------------------------------------------------------------------


def _spec(name: str) -> ExperimentSpec:
    import importlib

    return importlib.import_module(STRESS_MODULES[name]).SPEC


def seed_gaps(manifest: dict[str, Any]) -> list[str]:
    """Empty when the manifest records a seed, or explicitly says none applies (`seed: null` + `seed_status: not_applicable`)."""
    if manifest.get("seed_status") == "not_applicable" and manifest.get("seed") is None and "seed" in manifest:
        return []
    if any(manifest.get(k) is not None for k in SEED_KEYS):
        return []
    return ["no seed recorded in the dataset manifest (and no explicit seed_status: not_applicable)"]


def stress_dataset_checks(stress_root: Path = STRESS_DIR) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for name in STRESS_MODULES:
        spec = _spec(name)
        try:
            rows, manifest = load_frozen_dataset(stress_root / name, spec.dataset_name)  # verifies sha256 + unique variant_id + required fields
        except Exception as exc:  # noqa: BLE001 - report any failure as a failed check
            out.append(check(f"stress dataset {name}", "fail", str(exc)))
            continue
        problems = []
        if any(not r.get("source_example_id") for r in rows):
            problems.append("row without source_example_id")
        if len({r["variant_id"] for r in rows}) != len(rows):
            problems.append("duplicate variant_id")
        if manifest["n_rows"] != len(rows):
            problems.append("manifest n_rows mismatch")
        rebuilt, _ = spec.build()
        if json.dumps(rebuilt, sort_keys=True, default=str) != json.dumps(rows, sort_keys=True, default=str):
            problems.append("rebuilding from code does not reproduce the frozen rows")
        gaps = seed_gaps(manifest)
        detail = f"{len(rows)} rows, {manifest['n_source_examples']} source ids, sha256 {manifest['sha256'][:12]}, deterministic rebuild {'ok' if not problems else 'FAILED'}"
        out.append(check(f"stress dataset {name}", "fail" if problems else "pass", "; ".join(problems) or detail))
        if gaps:
            out.append(check(f"stress dataset {name}: seed/config recorded in its manifest", "warn", gaps[0] + " (the frozen rows are unaffected; the seed constants live in code and are recorded in the Phase II manifest)"))
    return out


class _FakeAnswer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice, self.probabilities = choice, probabilities


class _FakeUsage:
    input_tokens, output_tokens = 100, 5


class _FakeResponse:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choices = {"intent": _FakeAnswer(choice, probabilities)}
        self.model, self.usage, self.request_id = "fake-offline", _FakeUsage(), "req-fake"


class FakeOfflineClient:
    """Deterministic stand-in that never touches the network: picks a candidate from a hash of the state."""

    def __init__(self) -> None:
        self.calls = 0

    async def system_one(self, state, questions):
        self.calls += 1
        candidates = list(questions["intent"].criteria)
        h = int(hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8], 16)
        pick = candidates[h % len(candidates)]
        probs = {c: 0.02 for c in candidates}
        probs[pick] = round(0.4 + (h % 60) / 100, 2)
        return _FakeResponse(pick, probs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def pipeline_selfcheck(stress_root: Path = STRESS_DIR) -> list[dict[str, str]]:
    """build -> report -> run -> analyze for every experiment, with a FAKE client. Proves the schemas line up offline."""
    out: list[dict[str, str]] = []
    for name in STRESS_MODULES:
        spec = _spec(name)
        try:
            rows, _ = load_frozen_dataset(stress_root / name, spec.dataset_name)
            with tempfile.TemporaryDirectory() as tmp:
                results_dir = Path(tmp)
                fake = FakeOfflineClient()
                paths = asyncio.run(execute_live(spec.name, rows, f"stress:{spec.name}", ["fake"], results_dir, len(rows), 8, spec.state_builder, client_factory=lambda s, f=fake: (f, "fake")))
                results = read_results(results_dir)
                tables, summary = spec.analyze(results)
            assert len(results) == len(rows) and fake.calls == len(rows) and paths
            assert results["source_example_id"].notna().all(), "source_example_id lost in results"
            out.append(check(f"offline pipeline {name} (frozen dataset -> fake client -> analysis)", "pass", f"{len(rows)} rows -> {len(tables)} tables, no network"))
        except Exception as exc:  # noqa: BLE001
            out.append(check(f"offline pipeline {name}", "fail", f"{type(exc).__name__}: {exc}"))
    return out


# --- 4. live reproduction plan -------------------------------------------------------------------


def live_reproduction_plan(stress_root: Path = STRESS_DIR, baseline_dir: Path = BASELINE_DIR) -> dict[str, Any]:
    from evaluation.pricing import PRICING

    canonical = pd.read_parquet(baseline_dir / CANONICAL_NAME)
    tokens = {p: (float(g["input_tokens"].mean()), float(g["output_tokens"].mean())) for p, g in canonical.groupby("provider")}
    jev_key, llm_key = next(k for k in PRICING if k.startswith("jev")), "gpt-4o-mini"

    def per_call(model: str, provider: str) -> float:
        ti, to = tokens[provider]
        return ti / 1e6 * PRICING[model].input_per_million_usd + to / 1e6 * PRICING[model].output_per_million_usd

    datasets, jev_total = {}, 0
    for name in STRESS_MODULES:
        m = json.loads((stress_root / name / "dataset_manifest.json").read_text())
        n = m["n_rows"]
        cli = f"uv run python src/phase2/{STRESS_CLI[name]}"
        datasets[name] = {
            "rows": n,
            "jev_calls": n,
            "reference_llm_calls_optional": n,
            "commands": {"dry_run_no_api_calls": f"{cli} report", "live_jev_MAKES_API_CALLS": f"{cli} run --providers jev --approve-calls {n}", "live_jev_and_reference_llm_MAKES_API_CALLS": f"{cli} run --providers jev,openai:gpt-4o-mini --approve-calls {2 * n}", "analyze_offline": f"{cli} analyze"},
            "estimated_jev_cost_usd": round(n * per_call(jev_key, "jev"), 4),
            "estimated_reference_llm_cost_usd": round(n * per_call(llm_key, llm_key), 4),
        }
        jev_total += n
    return {
        "jev_calls_total": jev_total,
        "reference_llm_calls_total_optional": jev_total,
        "estimated_jev_cost_usd_total": round(sum(d["estimated_jev_cost_usd"] for d in datasets.values()), 4),
        "estimated_reference_llm_cost_usd_total": round(sum(d["estimated_reference_llm_cost_usd"] for d in datasets.values()), 4),
        "cost_basis": f"mean recorded Phase I tokens/call x src/evaluation/pricing.py rates ({jev_key}: owner notes, unverified; {llm_key}: OpenAI page read 2026-09-23). ESTIMATES; context-heavy datasets cost more.",
        "safety": "each `run` refuses unless --approve-calls equals the exact expected call count; dry runs (`report`) construct no client",
        "datasets": datasets,
    }


def render_plan_markdown(plan: dict[str, Any]) -> str:
    lines = ["# Phase II live-reproduction plan", "", "Generated by `uv run python src/phase2/verify_phase2.py plan --markdown docs/phase2-live-reproduction.md`. Nothing in this file has been executed.", "",
             f"- Jev calls to reproduce every Phase II live experiment: **{plan['jev_calls_total']}**",
             f"- Optional reference-LLM calls (same datasets): **{plan['reference_llm_calls_total_optional']}**",
             f"- Estimated cost: Jev ${plan['estimated_jev_cost_usd_total']}, reference LLM ${plan['estimated_reference_llm_cost_usd_total']}",
             f"- Cost basis: {plan['cost_basis']}", f"- Safety: {plan['safety']}", "", "Each experiment: run the dry run first (no API calls), then the live command (MAKES API CALLS), then analyze (offline).", ""]
    for name, d in plan["datasets"].items():
        lines += [f"## {name} ({d['rows']} rows)", "", "```bash", f"# dry run, no API calls\n{d['commands']['dry_run_no_api_calls']}", f"# LIVE: {d['jev_calls']} Jev calls (est. ${d['estimated_jev_cost_usd']})\n{d['commands']['live_jev_MAKES_API_CALLS']}",
                  f"# LIVE, optional reference LLM too: {2 * d['rows']} calls total (est. LLM ${d['estimated_reference_llm_cost_usd']})\n{d['commands']['live_jev_and_reference_llm_MAKES_API_CALLS']}", f"# offline analysis of recorded results\n{d['commands']['analyze_offline']}", "```", ""]
    return "\n".join(lines)


# --- 5. Phase II manifest ------------------------------------------------------------------------


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dir_hashes(directory: Path, names: tuple[str, ...] | None = None) -> dict[str, str]:
    files = [directory / n for n in names] if names else sorted(p for p in directory.glob("*") if p.is_file())
    return {p.name: sha256_file(p) for p in files if p.exists()}


def build_phase2_manifest(baseline_dir: Path = BASELINE_DIR, existing_root: Path = PHASE2_RESULTS_DIR, stress_root: Path = STRESS_DIR) -> dict[str, Any]:
    import importlib
    from datetime import UTC, datetime

    base = load_manifest(baseline_dir / MANIFEST_NAME)
    modules = sorted(Path("src/phase2").glob("*.py"))
    code_constants: dict[str, Any] = {}
    for label, mod, names in (("stress", "phase2.stress", ("SOURCE_SEED",)), ("ambiguity", "phase2.ambiguity", ("N_SOURCES",)), ("noise", "phase2.noise", ("NOISE_SEED", "N_SOURCES")), ("context_pollution", "phase2.context_pollution", ("CONTEXT_SEED", "CONTEXT_BUDGET_CHARS")),
                                ("ood", "phase2.ood", ("OOD_SEED", "FALSE_CONFIDENCE_THRESHOLDS")), ("failures", "phase2.failures", ("HIGH_CONFIDENCE_THRESHOLD", "LOW_CONFIDENCE_THRESHOLD")), ("cascade", "phase2.cascade", ("DEFAULT_THRESHOLDS",)),
                                ("full_cascade", "phase2.full_cascade", ("DEFAULT_JEV_THRESHOLDS", "DEFAULT_LLM_THRESHOLDS")), ("selective", "phase2.selective", ("STANDARD_THRESHOLDS",)), ("calibration", "phase2.calibration", ("N_BINS",))):
        m = importlib.import_module(mod)
        code_constants[label] = {n: getattr(m, n) for n in names}
    stress = {}
    for name in STRESS_MODULES:
        m = json.loads((stress_root / name / "dataset_manifest.json").read_text())
        rows, _ = load_frozen_dataset(stress_root / name)
        stress[name] = {"path": str(stress_root / name), "dataset_sha256": m["sha256"], "manifest_sha256": sha256_file(stress_root / name / "dataset_manifest.json"), "n_rows": m["n_rows"], "n_source_examples": m["n_source_examples"],
                        "source_example_ids_sha256": _sha_text("\n".join(sorted({r["source_example_id"] for r in rows}))), "seeds_stored_in_manifest": {k: m[k] for k in (*SEED_KEYS, "seed_status") if k in m} or NOT_RECORDED, "review_status": m.get("review_status")}
    canonical = pd.read_parquet(baseline_dir / CANONICAL_NAME)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "uncommitted_changes": bool(_git("status", "--porcelain"))},
        "historical_metadata": {"note": "copied from the frozen baseline manifest; anything not recorded at run time stays not_recorded and is NEVER back-filled with current values", **base["historical_metadata"]},
        "environment_at_manifest": {"note": "current values, NOT the values Phase I or any earlier run used", "python": sys.version.split()[0], "sdk": {n: metadata.version(n) for n in ("typesafe-sdk", "system-one-adapter")}},
        "providers": {"primary": "jev", "reference_llm": "gpt-4o-mini", "labels_in_canonical_results": sorted(canonical["provider"].unique()), "jev_model_version": NOT_RECORDED},
        "dataset_revisions": {"bitext": NOT_RECORDED, "massive": NOT_RECORDED, "stress_datasets": "content-pinned by the sha256 values below"},
        "phase1_result_files_sha256": {n: i["sha256"] for n, i in base["files"].items()},
        "baseline": {"manifest_sha256": sha256_file(baseline_dir / MANIFEST_NAME), "canonical_results_sha256": sha256_file(baseline_dir / CANONICAL_NAME), "canonical": base["canonical"]},
        "stress_datasets": stress,
        "sample_files_sha256": base["samples"],
        "experiment_code_sha256": {p.name: sha256_file(p) for p in modules},
        "seed_and_threshold_constants_in_code": code_constants,
        "config_hashes": {"pyproject.toml": sha256_file(Path("pyproject.toml")), "uv.lock": sha256_file(Path("uv.lock")), "src/evaluation/pricing.py": sha256_file(Path("src/evaluation/pricing.py"))},
        "analysis_outputs_sha256": {name: _dir_hashes(existing_root / name, names) for name, names in ANALYSIS_ARTIFACTS.items() if (existing_root / name).exists()},
        "live_stress_results": {"recorded": sorted(p.parent.name for p in existing_root.glob("*/raw_results_*.parquet")) or "none: live experiments have not been run"},
    }


def write_phase2_manifest(path: Path = MANIFEST_OUT, **kwargs: Any) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    manifest = build_phase2_manifest(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str, sort_keys=True))
    return manifest
