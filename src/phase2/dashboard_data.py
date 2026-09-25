"""Pure data loading and calculation for the Phase II dashboard tabs (no Streamlit, no model calls).

The dashboard READS precomputed artifacts; nothing here calls Jev or an LLM, and no headline number is
hard-coded (every figure comes from the files on disk). The results root, the frozen baseline and the stress datasets can be redirected with the
JEV_PHASE2_RESULTS_DIR, JEV_BASELINE_DIR and JEV_STRESS_DIR environment variables (used by tests).

Provider labels always distinguish Jev (the primary model) from the reference LLM.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME
from phase2.cascade import build_matched
from phase2.full_cascade import CascadeConfig, NoRules, attach_text, simulate_full, summarize_full


def phase2_root() -> Path:
    return Path(os.environ.get("JEV_PHASE2_RESULTS_DIR", "data/results/phase2"))


def baseline_dir() -> Path:
    return Path(os.environ.get("JEV_BASELINE_DIR", str(BASELINE_DIR)))


def stress_root() -> Path:
    return Path(os.environ.get("JEV_STRESS_DIR", "data/stress"))


# --- generic loaders -----------------------------------------------------------------------------


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_jsonl(path: Path) -> pd.DataFrame | None:
    """Strict JSON lines (a bare NaN would raise) -> DataFrame; None if missing/empty."""
    if not path.exists():
        return None

    def reject(token: str):
        raise ValueError(f"non-strict JSON constant {token} in {path}")

    rows = [json.loads(line, parse_constant=reject) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows) if rows else None


def provider_label(name: str) -> str:
    return "Jev (primary)" if name == "jev" else f"{name} (reference LLM)"


def dataset_info(name: str) -> dict[str, Any] | None:
    """Manifest of a frozen stress dataset (rows, sources), or None if it has not been built."""
    return load_json(stress_root() / name / "dataset_manifest.json")


# --- calibration ---------------------------------------------------------------------------------

GROUP_COLS = ["run_id", "experiment", "provider", "dataset", "num_choices"]


def group_label(row: pd.Series) -> str:
    parts = [provider_label(row["provider"]) if pd.notna(row.get("provider")) else "all providers"]
    for col in ("dataset", "experiment", "run_id", "locale"):
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}={row[col]}")
    if "num_choices" in row and pd.notna(row["num_choices"]):
        parts.append(f"K={int(row['num_choices'])}")
    return " · ".join(parts)


def group_options(summary: pd.DataFrame, level: str) -> dict[str, pd.Series]:
    """label -> summary row for one analysis level. Jev groups first."""
    sub = summary[summary["level"] == level]
    rows = sorted(sub.iterrows(), key=lambda kv: (kv[1].get("provider") != "jev", str(kv[1].get("provider")), group_label(kv[1])))
    return {group_label(r): r for _, r in rows}


def bins_for_group(bins: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """The 10 reliability bins belonging to one summary row (matched on every identifying column)."""
    mask = bins["level"] == row["level"]
    for col in GROUP_COLS:
        if col not in bins.columns:
            continue
        mask &= bins[col].isna() if pd.isna(row.get(col)) else (bins[col] == row[col])
    return bins[mask].sort_values("bin_index")


def sparse_bin_warnings(group_bins: pd.DataFrame, min_n: int = 30) -> list[str]:
    small = group_bins[(group_bins["sample_count"] > 0) & (group_bins["sample_count"] < min_n)]
    return [f"{r.bin_label}: n={int(r.sample_count)}" for r in small.itertuples()]


# --- risk / coverage -----------------------------------------------------------------------------

RISK_KEY_COLS = ["run_id", "experiment", "provider", "dataset", "num_choices", "locale"]


def risk_curve(table: pd.DataFrame, row_key: dict[str, Any], level: str) -> pd.DataFrame:
    mask = table["level"] == level
    for col in RISK_KEY_COLS:
        if col in table.columns:
            mask &= table[col].isna() if pd.isna(row_key.get(col)) else (table[col] == row_key[col])
    return table[mask].sort_values("threshold")


def risk_at(curve: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Coverage, accepted accuracy and risk at one threshold. Accuracy/risk are None when nothing is accepted."""
    hit = curve[(curve["threshold"] - threshold).abs() < 1e-9]
    if hit.empty:
        raise ValueError(f"threshold {threshold} is not on this curve")
    r = hit.iloc[0]
    none = lambda v: None if pd.isna(v) else float(v)  # noqa: E731
    return {"threshold": float(r["threshold"]), "total": int(r["total"]), "accepted": int(r["accepted"]), "rejected": int(r["rejected"]),
            "coverage": none(r["coverage"]), "accepted_accuracy": none(r["accepted_accuracy"]), "risk": none(r["risk"])}


# --- failures ------------------------------------------------------------------------------------


def filter_failures(df: pd.DataFrame, *, experiments: list[str] | None = None, providers: list[str] | None = None, locales: list[str] | None = None, tags: list[str] | None = None,
                    conf_range: tuple[float, float] = (0.0, 1.0), ground_truth: list[str] | None = None, prediction: list[str] | None = None) -> pd.DataFrame:
    """AND across filters; within `tags` a failure matches if it carries ANY selected tag. Empty/None = no filter."""
    out = df
    if experiments:
        out = out[out["experiment"].isin(experiments)]
    if providers:
        out = out[out["provider"].isin(providers)]
    if locales:
        out = out[out["locale"].isin(locales)]
    if tags:
        out = out[out["failure_tags"].map(lambda t: any(x in t for x in tags))]
    lo, hi = conf_range
    if (lo, hi) != (0.0, 1.0):
        out = out[out["confidence"].between(lo, hi)]  # failures without a confidence are dropped only when a range is chosen
    if ground_truth:
        out = out[out["ground_truth"].isin(ground_truth)]
    if prediction:
        out = out[out["prediction"].isin(prediction)]
    return out


# --- cascade simulator ---------------------------------------------------------------------------


def load_matched(baseline_dir: Path = BASELINE_DIR) -> pd.DataFrame | None:
    """Matched Jev/reference-LLM requests from the frozen canonical baseline (precomputed; no live calls)."""
    path = baseline_dir / CANONICAL_NAME
    if not path.exists():
        return None
    canonical = pd.read_parquet(path)
    matched, _ = build_matched(canonical)
    return attach_text(matched, canonical)


def workloads(matched: pd.DataFrame) -> list[str]:
    return sorted(f"{e}|k={int(k)}" for e, k in matched[["experiment", "num_choices"]].drop_duplicates().itertuples(index=False))


def run_cascade_simulation(matched: pd.DataFrame, workload: str, jev_threshold: float, llm_enabled: bool, oracle_enabled: bool, llm_threshold: float | None = 0.5) -> dict[str, Any]:
    """Simulate the cascade on ONE workload (never pooled) with the current control settings. Labelled SIMULATED."""
    exp, k = workload.split("|k=")
    grp = matched[(matched["experiment"] == exp) & (matched["num_choices"] == int(k))]
    if grp.empty:
        raise ValueError(f"unknown workload {workload!r}")
    cfg = CascadeConfig(jev_threshold=jev_threshold, llm_enabled=llm_enabled, llm_threshold=llm_threshold, oracle_enabled=oracle_enabled)
    s = summarize_full(simulate_full(grp, cfg, NoRules()))
    s["workload"], s["config"] = workload, {"jev_threshold": jev_threshold, "llm_enabled": llm_enabled, "oracle_enabled": oracle_enabled, "llm_threshold": llm_threshold}
    s["simulated"] = True
    return s
