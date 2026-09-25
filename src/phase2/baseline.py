"""Phase II Step 1: freeze the Phase I baseline into a reproducible manifest.

Copies (never moves or rewrites) Phase I result files into
`data/results/phase2/baseline/`, and records a manifest describing exactly what
was run: dataset revisions, example IDs, seeds, SDK/model versions, question
config, concurrency and a timestamp. Makes zero API calls.

Phase I outputs are only ever read. Frozen copies are made read-only, and
`create_baseline` refuses to overwrite an existing baseline.

Raw frozen copies stay byte-for-byte identical to Phase I. A derived canonical
view (`canonical_results.parquet`) collapses repeated predictions; all Phase II
analysis should use it, with the raw files kept for provenance.

In Phase I, `example_id` is the source example ID; Phase II transformed
examples must carry it forward as `source_example_id`.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from dataset_loaders.bitext import DATASET_NAME as BITEXT_DATASET
from dataset_loaders.bitext import DEFAULT_SEED as BITEXT_SEED
from dataset_loaders.bitext import SAMPLE_SIZES as BITEXT_SAMPLE_SIZES
from dataset_loaders.massive import ALIGNED_SAMPLE_SIZES, LOCALES
from dataset_loaders.massive import DEFAULT_SEED as MASSIVE_SEED
from evaluation.metrics import compute_metrics_by_group
from models.experiment import ExperimentConfig

SCHEMA_VERSION = 1
PHASE1_RESULTS_DIR = Path("data/results")
SAMPLES_DIR = Path("data/samples")
BASELINE_DIR = PHASE1_RESULTS_DIR / "phase2" / "baseline"
MANIFEST_NAME = "baseline_manifest.json"
MASSIVE_DATASET = "AmazonScience/massive"

CANONICAL_NAME = "canonical_results.parquet"
DEDUP_REPORT_NAME = "canonical_dedup_report.csv"
NOT_RECORDED = "not_recorded"
# One prediction is identified by: which run, which provider, which example, which locale.
IDENTITY_KEY = ["run_id", "provider", "example_id", "locale"]
DEDUP_RULE = (
    "Identity = (run_id, provider, example_id, locale). For every identity with >1 row, keep only the LATEST "
    "row (highest raw row position) and drop the earlier ones. The rule uses row order only, never the "
    "prediction or its correctness, so it cannot bias the canonical set. Whether repeated calls agreed is "
    "recorded separately in canonical_dedup_report.csv as evidence of decision stability."
)

# Choice question built by evaluation.runner.SystemOneEvaluator in the CODE AT FREEZE TIME -- not a Phase I
# record (a test asserts the instructions string still appears there).
QUESTION_CONFIG = {
    "question_type": "Choice",
    "question_key": "intent",
    "instructions": "Select the intent that best represents the user's request.",
    "criteria": "one entry per example candidate, each with value None",
}
# Phase I never persisted concurrency. The CLI default is noted separately and is NOT a historical value.
CONCURRENCY = {"value": None, "status": NOT_RECORDED, "current_cli_default": 5}

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


class BaselineError(Exception):
    """Baseline could not be created or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_cache_revision(dataset_id: str, cache: Path = HF_CACHE) -> dict[str, Any]:
    """What the local HF cache holds now (no network). Environment info only, never a historical revision."""
    base = cache / f"datasets--{dataset_id.replace('/', '--')}"
    ref = base / "refs" / "main"
    snapshots = sorted(p.name for p in (base / "snapshots").glob("*")) if (base / "snapshots").exists() else []
    return {
        "hf_cache_refs_main_at_freeze": ref.read_text().strip() if ref.exists() else None,
        "cached_snapshots_at_freeze": snapshots,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _candidate_configs(run_df: pd.DataFrame) -> list[tuple[str, str]]:
    """(config name, dataset string) guesses for a run, from the rows it holds."""
    experiment = str(run_df["experiment"].iloc[0])
    n_unique = run_df["example_id"].nunique()
    if experiment.startswith("choice_scaling_k"):
        return [("choice_scaling", f"massive:choice_scaling:k{experiment.removeprefix('choice_scaling_k')}")]
    if experiment == "bitext_hard_choice":
        return [(experiment, f"bitext:{name}") for name, size in BITEXT_SAMPLE_SIZES.items() if size == n_unique]
    if experiment == "multilingual":
        present = set(run_df["locale"].dropna())
        locale_orders = [[loc for loc in LOCALES if loc in present], sorted(present)]
        return [(experiment, f"massive:{name}:{'+'.join(order)}") for name in ALIGNED_SAMPLE_SIZES for order in locale_orders]
    return []


def recover_config(run_df: pd.DataFrame) -> dict[str, Any] | None:
    """Rebuild the ExperimentConfig behind a run_id by matching its deterministic hash."""
    run_id = str(run_df["run_id"].iloc[0])
    providers = tuple(sorted(run_df["provider"].unique()))
    for name, dataset in _candidate_configs(run_df):
        for seed in (0, BITEXT_SEED, MASSIVE_SEED):
            if ExperimentConfig(name=name, dataset=dataset, providers=providers, seed=seed).run_id == run_id:
                return {"name": name, "dataset": dataset, "providers": list(providers), "seed": seed}
    return None


REPORT_COLUMNS = [
    *IDENTITY_KEY,
    "copies",
    "earlier_prediction",
    "later_prediction",
    "earlier_confidence",
    "later_confidence",
    "predictions_agree",
    "earlier_request_id",
    "later_request_id",
    "earlier_latency_ms",
    "later_latency_ms",
    "raw_rows",
    "selected_raw_row",
]


def build_canonical(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the canonical view and a per-identity report from raw Phase I rows.

    For each duplicated identity the latest raw row is kept, regardless of what it predicted.
    `canonical_df` keeps a `raw_row` column pointing back to the row's position in the raw file.
    `report_df` has one row per duplicated identity, including whether its repeated calls agreed.
    """
    df = raw.reset_index(drop=True).rename_axis("raw_row").reset_index()
    group_sizes = df.groupby(IDENTITY_KEY, dropna=False)["raw_row"].transform("size")

    report = []
    for key, grp in df[group_sizes > 1].groupby(IDENTITY_KEY, dropna=False, sort=True):
        grp = grp.sort_values("raw_row")
        first, last = grp.iloc[0], grp.iloc[-1]
        report.append(
            {
                **dict(zip(IDENTITY_KEY, key, strict=True)),
                "copies": len(grp),
                "earlier_prediction": first["prediction"],
                "later_prediction": last["prediction"],
                "earlier_confidence": first["confidence"],
                "later_confidence": last["confidence"],
                "predictions_agree": grp["prediction"].nunique(dropna=False) == 1,
                "earlier_request_id": first["request_id"],
                "later_request_id": last["request_id"],
                "earlier_latency_ms": first["latency_ms"],
                "later_latency_ms": last["latency_ms"],
                "raw_rows": ";".join(map(str, grp["raw_row"])),
                "selected_raw_row": int(last["raw_row"]),
            }
        )
    report_df = pd.DataFrame(report, columns=REPORT_COLUMNS)
    canonical = df[df["raw_row"].isin(set(df["raw_row"]) - set(_dropped_rows(df, group_sizes)))].reset_index(drop=True)
    return canonical, report_df


def _dropped_rows(df: pd.DataFrame, group_sizes: pd.Series) -> list[int]:
    """Every row of a duplicated identity except its latest one."""
    dup = df[group_sizes > 1]
    latest = dup.groupby(IDENTITY_KEY, dropna=False)["raw_row"].transform("max")
    return dup.loc[dup["raw_row"] != latest, "raw_row"].tolist()


def dedup_summary(raw_rows: int, canonical_rows: int, report: pd.DataFrame) -> dict[str, Any]:
    agreeing = int(report["predictions_agree"].sum())
    return {
        "rule": DEDUP_RULE,
        "identity_key": IDENTITY_KEY,
        "raw_rows": raw_rows,
        "canonical_rows": canonical_rows,
        "rows_removed": raw_rows - canonical_rows,
        "duplicated_identities": len(report),
        "agreeing_repeated_calls": agreeing,
        "disagreeing_repeated_calls": len(report) - agreeing,
        "repeat_prediction_agreement": agreeing / len(report) if len(report) else None,
        "note": "Use canonical_results.parquet for all Phase II analysis; the raw frozen files are for provenance only. Disagreements are evidence of decision instability, not removed rows.",
    }


def _describe_runs(raw: pd.DataFrame, canonical: pd.DataFrame, report: pd.DataFrame) -> list[dict[str, Any]]:
    runs = []
    for run_id, run_df in raw.groupby("run_id", sort=True):
        canon_df = canonical[canonical["run_id"] == run_id]
        run_report = report[report["run_id"] == run_id]
        ids = list(dict.fromkeys(run_df["example_id"]))
        runs.append(
            {
                "run_id": run_id,
                "experiment": str(run_df["experiment"].iloc[0]),
                "providers": sorted(run_df["provider"].unique()),
                "config": recover_config(run_df),
                "raw_rows": len(run_df),
                "canonical_rows": len(canon_df),
                "duplicated_identities": len(run_report),
                "rows_removed": len(run_df) - len(canon_df),
                "disagreeing_repeated_calls": int((~run_report["predictions_agree"]).sum()),
                "n_unique_examples": len(ids),
                "example_ids": ids,
                "example_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "raw_metrics_by_provider": compute_metrics_by_group(run_df, "provider"),
                "canonical_metrics_by_provider": compute_metrics_by_group(canon_df, "provider") if len(canon_df) else {},
            }
        )
    return runs


def _describe_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".parquet":
        info["rows"] = len(pd.read_parquet(path))
    return info


def _phase1_files(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.iterdir() if p.is_file() and p.name != ".gitkeep" and not p.name.endswith(".tmp"))


def create_baseline(
    results_dir: Path = PHASE1_RESULTS_DIR,
    samples_dir: Path = SAMPLES_DIR,
    out_dir: Path = BASELINE_DIR,
) -> dict[str, Any]:
    """Copy Phase I outputs into `out_dir` and write the manifest. Refuses to overwrite."""
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise BaselineError(f"Baseline already exists at {manifest_path}; refusing to overwrite. Verify it, or choose another --out.")
    sources = _phase1_files(results_dir)
    if not sources:
        raise BaselineError(f"No Phase I result files found in {results_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    for src in sources:
        dest = out_dir / src.name
        if dest.exists():
            raise BaselineError(f"{dest} already exists; refusing to overwrite.")
        shutil.copy2(src, dest)
        dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        files[src.name] = {"source": str(src), **_describe_file(dest)}

    samples = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(samples_dir.glob("*")) if p.is_file() and p.name != ".gitkeep"}

    derived: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []
    canonical_summary: dict[str, Any] | None = None
    frozen_results = out_dir / "results.parquet"
    if frozen_results.exists():
        raw = pd.read_parquet(frozen_results)
        canonical, report = build_canonical(raw)
        for name, write in ((CANONICAL_NAME, lambda d: canonical.to_parquet(d, index=False)), (DEDUP_REPORT_NAME, lambda d: report.to_csv(d, index=False))):
            dest = out_dir / name
            if dest.exists():
                raise BaselineError(f"{dest} already exists; refusing to overwrite.")
            write(dest)
            dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            derived[name] = {"derived_from": "results.parquet", **_describe_file(dest)}
        runs = _describe_runs(raw, canonical, report)
        canonical_summary = dedup_summary(len(raw), len(canonical), report)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "historical_metadata": {
            "jev_model_version": {"value": None, "status": NOT_RECORDED},
            "sdk_version": {"value": None, "status": NOT_RECORDED},
            "concurrency": CONCURRENCY,
            "dataset_revision": {
                "bitext": {"hf_dataset": BITEXT_DATASET, "value": None, "status": NOT_RECORDED},
                "massive": {"hf_dataset": MASSIVE_DATASET, "value": None, "status": NOT_RECORDED},
            },
            "sample_seeds": {"value": None, "status": NOT_RECORDED, "current_code_defaults": {"bitext": BITEXT_SEED, "massive": MASSIVE_SEED}},
            "experiment_config": "recovered per run under runs[].config by matching the deterministic run_id hash (verified); null if no match",
            "question_config": {"value": None, "status": NOT_RECORDED},
        },
        "environment_at_freeze": {
            "note": "Current values when this baseline was created. NOT the values Phase I ran with.",
            "current_code_at_freeze": {"question_config": QUESTION_CONFIG},
            "python": sys.version.split()[0],
            "sdk": {name: metadata.version(name) for name in ("typesafe-sdk", "system-one-adapter")},
            "jev_model_setting": settings.jev_model,
            "hf_cache": {"bitext": hf_cache_revision(BITEXT_DATASET), "massive": hf_cache_revision(MASSIVE_DATASET)},
        },
        "canonical": canonical_summary,
        "samples": samples,
        "files": files,
        "derived_files": derived,
        "runs": runs,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return manifest


def load_manifest(path: Path = BASELINE_DIR / MANIFEST_NAME) -> dict[str, Any]:
    if not path.exists():
        raise BaselineError(f"No baseline manifest at {path}.")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(f"Unsupported manifest schema_version {manifest.get('schema_version')!r} (expected {SCHEMA_VERSION}).")
    return manifest


def verify_baseline(
    manifest_path: Path = BASELINE_DIR / MANIFEST_NAME,
    samples_dir: Path = SAMPLES_DIR,
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors: the frozen baseline is damaged. Warnings: Phase I sources changed since."""
    manifest = load_manifest(manifest_path)
    base = manifest_path.parent
    errors: list[str] = []
    warnings: list[str] = []

    for name, info in manifest["files"].items():
        frozen = base / name
        if not frozen.exists():
            errors.append(f"frozen file missing: {name}")
            continue
        if sha256_file(frozen) != info["sha256"]:
            errors.append(f"frozen file modified: {name}")
        source = Path(info["source"])
        if not source.exists():
            warnings.append(f"Phase I source no longer present: {source}")
        elif sha256_file(source) != info["sha256"]:
            warnings.append(f"Phase I source changed since freeze: {source}")

    for name, info in manifest["derived_files"].items():
        derived = base / name
        if not derived.exists():
            errors.append(f"derived file missing: {name}")
        elif sha256_file(derived) != info["sha256"]:
            errors.append(f"derived file modified: {name}")

    frozen_results = base / "results.parquet"
    if frozen_results.exists() and not errors:
        raw = pd.read_parquet(frozen_results)
        canonical, report = build_canonical(raw)
        if not canonical.equals(pd.read_parquet(base / CANONICAL_NAME)):
            errors.append("canonical view does not match a fresh derivation from the frozen raw file")
        current = {r["run_id"]: r for r in _describe_runs(raw, canonical, report)}
        for run in manifest["runs"]:
            found = current.get(run["run_id"])
            if found is None or found["example_ids_sha256"] != run["example_ids_sha256"]:
                errors.append(f"example IDs differ for run {run['run_id']}")

    for name, info in manifest["samples"].items():
        sample = samples_dir / name
        if not sample.exists():
            warnings.append(f"sample file no longer present: {sample}")
        elif sha256_file(sample) != info["sha256"]:
            warnings.append(f"sample file changed since freeze: {sample}")
    return errors, warnings
