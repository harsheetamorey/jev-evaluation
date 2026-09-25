"""Phase II Step 14: Jev -> LLM cascade, as an OFFLINE simulation (no model calls).

    Jev -> confidence >= threshold ? accept Jev : fall back to the LLM

Data coverage first: a cascade can only be simulated on examples that have BOTH a Jev and a reference
LLM prediction for the same example, locale AND candidate contract (identical candidate set). Examples
missing either side are reported, never filled in. If LLM predictions are missing for important Phase II
datasets, this module reports exactly how many extra calls are needed and does NOT make them.

Simulation assumptions (all stated in the outputs):
  * routing: Jev accepted iff Jev confidence >= threshold (inclusive); otherwise the request falls back.
  * latency: SEQUENTIAL. accepted path = Jev latency; fallback path = Jev latency + LLM latency. These are
    sums of independently MEASURED per-call latencies, i.e. a SIMULATED path latency, not a measured cascade.
    p50/p95 are computed over the per-request path latencies, never by averaging provider medians.
  * cost: Jev runs on EVERY request; the LLM runs only on fallbacks. Costs are the `estimated_cost_usd`
    values recorded at run time (token counts x src/evaluation/pricing.py rates), so ALL costs here are
    ESTIMATES, not billed amounts. Rows without a recorded cost make that total unavailable (never zero).
  * per-example accuracy uses the canonical baseline's latest measurement (Step 1 rule).
No "best threshold" is chosen; results are trade-offs.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.pricing import PRICING
from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.calibration import dedup_measurements
from phase2.stress import PHASE2_RESULTS_DIR, STRESS_DIR

CASCADE_DIR = PHASE2_RESULTS_DIR / "cascade"
JEV = "jev"
LLM = "gpt-4o-mini"
DEFAULT_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
SWEEP_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(50)]
KEY = ["experiment", "example_id", "locale"]
COST_PROVENANCE = "estimated_cost_usd recorded at run time = token counts x src/evaluation/pricing.py rates (jev: owner notes, unverified; gpt-4o-mini: OpenAI page read 2026-09-23). Estimates, not billing."
LATENCY_PROVENANCE = "simulated path latency: sum of independently measured per-call latencies (sequential cascade); not a measured cascade"
STRESS_DATASETS = ("ambiguity", "stability", "noise", "context_pollution", "context_relevance", "ood", "adversarial", "choice_overlap")
OUTPUTS = ("coverage_audit.csv", "missing_llm_report.json", "cascade_thresholds.csv", "cascade_per_example.csv", "cascade_summary.json")


def _cands(v: Any) -> tuple[str, ...]:
    return tuple(sorted(v))


def build_matched(canonical: pd.DataFrame, jev: str = JEV, llm: str = LLM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join Jev and LLM canonical rows on (experiment, example, locale) with an identical candidate contract.

    Returns (matched, coverage_audit). A pair is matched only if both sides produced a scored prediction.
    """
    df = dedup_measurements(canonical.assign(locale=canonical["locale"], candidates=canonical["candidates"].map(list)))
    df = df.assign(contract=df["candidates"].map(_cands), num_choices=df["candidates"].map(len))
    cols = ["prediction", "confidence", "correct", "latency_ms", "estimated_cost_usd"]
    a = df[df["provider"] == jev][[*KEY, "contract", "num_choices", "dataset", "ground_truth", *cols]].rename(columns={c: f"jev_{c}" for c in cols})
    b = df[df["provider"] == llm][[*KEY, "contract", *cols]].rename(columns={c: f"llm_{c}" for c in cols})
    a["_lk"], b["_lk"] = a["locale"].fillna("∅"), b["locale"].fillna("∅")
    keys = ["experiment", "example_id", "_lk"]
    both = a.merge(b.drop(columns="locale"), on=keys, how="outer", suffixes=("", "_llm"), indicator=True)
    both["locale"] = both["locale"].where(both["_lk"] != "∅", None) if "locale" in both else None
    audit_rows, matched_idx = [], []
    for exp, g in both.groupby("experiment", sort=True):
        in_both = g[g["_merge"] == "both"]
        contract_ok = in_both["contract"] == in_both["contract_llm"]
        scored = in_both["jev_correct"].notna() & in_both["llm_correct"].notna()
        ok = in_both[contract_ok & scored]
        matched_idx += ok.index.tolist()
        audit_rows.append(
            {
                "experiment": exp,
                "n_jev": int((g["_merge"] != "right_only").sum()),
                "n_llm": int((g["_merge"] != "left_only").sum()),
                "n_matched": len(ok),
                "n_jev_without_llm": int((g["_merge"] == "left_only").sum()),
                "n_llm_without_jev": int((g["_merge"] == "right_only").sum()),
                "n_contract_mismatch": int((~contract_ok).sum()),
                "n_unscored_pair_excluded": int((contract_ok & ~scored).sum()),
            }
        )
    matched = both.loc[matched_idx].drop(columns=["_merge", "_lk", "contract_llm"]).reset_index(drop=True)
    matched["dataset"] = matched["dataset"].fillna("unknown")
    return matched, pd.DataFrame(audit_rows)


def simulate(matched: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-example routing outcome at one threshold. Unresolvable fallbacks (no LLM prediction) stay unresolved."""
    conf = matched["jev_confidence"].round(6)
    accepted = matched["jev_prediction"].notna() & (conf >= threshold)
    llm_present = matched["llm_prediction"].notna()
    out = matched[[*KEY, "experiment", "num_choices", "ground_truth"]].copy() if "num_choices" in matched else matched[KEY].copy()
    out["threshold"] = threshold
    out["route"] = np.where(accepted, "jev", np.where(llm_present, "llm", "unresolved_missing_llm"))
    out["final_prediction"] = np.where(accepted, matched["jev_prediction"], np.where(llm_present, matched["llm_prediction"], None))
    out["final_correct"] = pd.Series(np.where(accepted, matched["jev_correct"], np.where(llm_present, matched["llm_correct"], None)), index=matched.index).astype(object)
    fallback = ~accepted
    out["path_latency_ms"] = matched["jev_latency_ms"] + np.where(fallback & llm_present, matched["llm_latency_ms"], 0.0)
    out["jev_cost_usd"] = matched["jev_estimated_cost_usd"]  # Jev runs on every request
    out["llm_cost_usd"] = np.where(fallback & llm_present, matched["llm_estimated_cost_usd"], 0.0)
    out["llm_cost_usd"] = np.where(fallback & llm_present & matched["llm_estimated_cost_usd"].isna(), np.nan, out["llm_cost_usd"])
    out["cost_usd"] = out["jev_cost_usd"] + out["llm_cost_usd"]
    return out


def summarize_threshold(sim: pd.DataFrame) -> dict[str, Any]:
    n = len(sim)
    resolved = sim[sim["route"] != "unresolved_missing_llm"]
    correct = resolved["final_correct"].astype(bool)
    jev_stage, llm_stage = sim[sim["route"] == "jev"], sim[sim["route"] == "llm"]
    lat = sim["path_latency_ms"]
    cost_ok = sim["cost_usd"].notna().all() and n > 0
    return {
        "threshold": float(sim["threshold"].iloc[0]) if n else None,
        "n_total": n,
        "n_handled_by_jev": len(jev_stage),
        "n_fell_back_to_llm": len(llm_stage),
        "n_unresolved_missing_llm": n - len(resolved),
        "pct_handled_by_jev": len(jev_stage) / n if n else None,
        "pct_fell_back_to_llm": len(llm_stage) / n if n else None,
        "final_accuracy": float(correct.mean()) if len(resolved) else None,
        "n_final_correct": int(correct.sum()),
        "jev_stage_errors": int((~jev_stage["final_correct"].astype(bool)).sum()),
        "llm_fallback_errors": int((~llm_stage["final_correct"].astype(bool)).sum()),
        "jev_stage_accuracy": float(jev_stage["final_correct"].astype(bool).mean()) if len(jev_stage) else None,
        "llm_fallback_accuracy": float(llm_stage["final_correct"].astype(bool).mean()) if len(llm_stage) else None,
        "path_latency_p50_ms": float(lat.quantile(0.5)) if n else None,
        "path_latency_p95_ms": float(lat.quantile(0.95)) if n else None,
        "total_cost_usd_estimated": float(sim["cost_usd"].sum()) if cost_ok else None,
        "cost_per_1000_requests_usd_estimated": float(sim["cost_usd"].sum() / n * 1000) if cost_ok else None,
        "n_missing_cost": int(sim["cost_usd"].isna().sum()),
    }


def reference_rows(matched: pd.DataFrame) -> dict[str, Any]:
    """Single-provider baselines on the SAME matched examples, for direct comparison with the cascade."""
    n = len(matched)
    out: dict[str, Any] = {"n_total": n}
    for name, pfx in (("jev_only", "jev"), ("llm_only", "llm")):
        cost = matched[f"{pfx}_estimated_cost_usd"]
        out[name] = {
            "accuracy": float(matched[f"{pfx}_correct"].astype(bool).mean()) if n else None,
            "latency_p50_ms": float(matched[f"{pfx}_latency_ms"].quantile(0.5)) if n else None,
            "latency_p95_ms": float(matched[f"{pfx}_latency_ms"].quantile(0.95)) if n else None,
            "cost_per_1000_requests_usd_estimated": float(cost.sum() / n * 1000) if n and cost.notna().all() else None,
        }
    return out


def threshold_table(matched: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for (exp, k), grp in matched.groupby(["experiment", "num_choices"], sort=True):
        for t in thresholds:
            rows.append({"experiment": exp, "num_choices": int(k), "dataset": grp["dataset"].iloc[0], **summarize_threshold(simulate(grp, t))})
    return pd.DataFrame(rows)


def estimate_missing_llm(canonical: pd.DataFrame, audit: pd.DataFrame, stress_dir: Path = STRESS_DIR) -> dict[str, Any]:
    """Extra LLM calls needed to cover gaps, with a labelled cost estimate. Reports only; makes no call."""
    llm = canonical[canonical["provider"] == LLM]
    tok_in, tok_out = float(llm["input_tokens"].mean()), float(llm["output_tokens"].mean())
    rate = PRICING[LLM]
    per_call_cost = tok_in / 1e6 * rate.input_per_million_usd + tok_out / 1e6 * rate.output_per_million_usd
    canonical_gap = int(audit["n_jev_without_llm"].sum())
    stress, total = {}, canonical_gap
    for name in STRESS_DATASETS:
        manifest = stress_dir / name / "dataset_manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        stress[name] = {"llm_calls_needed": m["n_rows"], "estimated_cost_usd": round(m["n_rows"] * per_call_cost, 4)}
        total += m["n_rows"]
    return {
        "canonical_examples_with_jev_but_no_llm": canonical_gap,
        "frozen_stress_datasets_needing_llm_predictions": stress,
        "total_additional_llm_calls_if_all_covered": total,
        "estimated_cost_usd_all": round(total * per_call_cost, 4),
        "estimation_basis": f"mean recorded {LLM} tokens/call from Phase I ({tok_in:.0f} in, {tok_out:.0f} out) x pricing.py rates; ESTIMATE, context-heavy datasets will cost more",
        "action": "NOT executed: no LLM call is made without explicit approval",
    }


def run_cascade(baseline_dir: Path = BASELINE_DIR, out_dir: Path = CASCADE_DIR, thresholds: list[float] = DEFAULT_THRESHOLDS, sweep: bool = False, verify: bool = True) -> dict[str, Any]:
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification: {errors}")
    existing = [str(out_dir / n) for n in OUTPUTS if (out_dir / n).exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing output(s): {existing}")
    source = baseline_dir / CANONICAL_NAME
    canonical = pd.read_parquet(source)
    matched, audit = build_matched(canonical)
    table = threshold_table(matched, sorted(set(thresholds) | (set(SWEEP_THRESHOLDS) if sweep else set())))
    per_example = pd.concat([simulate(g, t) for _, g in matched.groupby(["experiment", "num_choices"]) for t in thresholds], ignore_index=True)
    refs = {f"{exp}|k={int(k)}": reference_rows(g) for (exp, k), g in matched.groupby(["experiment", "num_choices"], sort=True)}
    missing = estimate_missing_llm(canonical, audit)
    summary = {
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "providers": {"primary": JEV, "fallback_reference": LLM},
        "n_matched_examples": len(matched),
        "thresholds": thresholds,
        "grouping": "per experiment (and its number of choices); workloads are not pooled",
        "same_population_references": refs,
        "assumptions": {"routing": "Jev accepted iff confidence >= threshold (inclusive); else LLM", "latency": LATENCY_PROVENANCE, "cost": COST_PROVENANCE},
        "no_best_threshold_selected": True,
        "interpretation": "not concluded here; results are trade-offs between accuracy, LLM fallback share, latency and estimated cost",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "coverage_audit.csv", index=False)
    (out_dir / "missing_llm_report.json").write_text(json.dumps(missing, indent=2))
    table.to_csv(out_dir / "cascade_thresholds.csv", index=False)
    per_example.to_csv(out_dir / "cascade_per_example.csv", index=False)
    (out_dir / "cascade_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
