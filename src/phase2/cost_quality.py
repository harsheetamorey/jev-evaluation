"""Phase II Step 16: cost x quality comparison across architectures (no model calls).

Architectures compared, per WORKLOAD (experiment + number of choices), never across workloads:

    rules_only            unavailable: the repo has no rules baseline (reported, not invented)
    jev_only              Jev on every request
    llm_only              the reference LLM on every request
    jev_to_llm            Jev, falling back to the LLM below a confidence threshold (Step 14)
    rules_jev_llm         identical to jev_to_llm while rules coverage is 0 (labelled degenerate)
    rules_jev_llm_oracle  the full cascade ending in an ORACLE (Step 15). Its accuracy includes ground-truth
                          answers by construction and its cost/latency are only partly defined.

Comparability rule: every architecture point in a workload is computed on the SAME set of requests; a
population fingerprint is stored and `assert_same_population` fails loudly if two points differ.

Provenance (stored per row):
  cost    : every cost is an ESTIMATE recorded at run time (token counts x src/evaluation/pricing.py rates);
            none is a billed amount. Totals are unavailable (never zero) if any needed cost is missing.
  latency : `measured` = one provider's recorded per-call latency; `simulated_sequential_path` = sum of measured
            per-call latencies along the cascade path (not a measured cascade). Oracle latency is NA.
No winner is chosen. `on_automated_cost_accuracy_frontier` is a DESCRIPTIVE flag (no other automated point is both
cheaper-or-equal and at-least-as-accurate, with one strictly better); it says nothing about which objective matters.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.cascade import COST_PROVENANCE, build_matched, simulate, summarize_threshold
from phase2.full_cascade import CascadeConfig, NoRules, attach_text, simulate_full, summarize_full
from phase2.stress import PHASE2_RESULTS_DIR

COST_QUALITY_DIR = PHASE2_RESULTS_DIR / "cost_quality"
OUTPUTS = ("architecture_frontier.csv", "cost_accuracy_points.csv", "frontier_summary.json")
JEV_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
ORACLE_LLM_THRESHOLDS = [0.5, 0.8]
LATENCY_MEASURED = "measured (one provider's recorded per-call latency)"
LATENCY_SIMULATED = "simulated_sequential_path (sum of measured per-call latencies along the cascade path; not a measured cascade)"
NO_RULES_REASON = "no rules baseline exists in this repository (src/clients/rules_client.py is empty)"
COLUMNS = ["workload", "experiment", "num_choices", "dataset", "architecture", "config", "n", "population_sha256", "accuracy", "n_correct", "cost_per_1000_requests_usd", "cost_complete", "cost_per_correct_decision_usd",
           "cost_provenance", "latency_p50_ms", "latency_p95_ms", "latency_n", "latency_provenance", "automation_coverage", "oracle_fallback_rate", "includes_oracle", "degenerate_reason", "status", "note"]


def population_fingerprint(matched: pd.DataFrame) -> str:
    keys = sorted(f"{r.experiment}|{r.example_id}|{'' if pd.isna(r.locale) else r.locale}" for r in matched.itertuples())
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def assert_same_population(points: list[dict[str, Any]]) -> None:
    """Every architecture point of ONE workload must be computed on the same request set."""
    if len({p["population_sha256"] for p in points}) > 1 or len({p["n"] for p in points}) > 1:
        raise ValueError("architecture points in one workload were computed on different populations; refusing to compare them")


def _cost_stats(total: float | None, n: int, n_correct: int) -> tuple[float | None, float | None]:
    if total is None or n == 0:
        return None, None
    return total / n * 1000, (total / n_correct if n_correct else None)


def _point(workload: dict[str, Any], pop_sha: str, architecture: str, config: str, **kw: Any) -> dict[str, Any]:
    row = {c: None for c in COLUMNS}
    row.update(workload)
    row.update({"architecture": architecture, "config": config, "population_sha256": pop_sha, "cost_provenance": COST_PROVENANCE, "includes_oracle": False, "status": "computed", **kw})
    return row


def single_provider_point(matched: pd.DataFrame, workload: dict[str, Any], pop_sha: str, prefix: str, name: str) -> dict[str, Any]:
    n = len(matched)
    cost = matched[f"{prefix}_estimated_cost_usd"]
    total = float(cost.sum()) if n and cost.notna().all() else None
    n_correct = int(matched[f"{prefix}_correct"].astype(bool).sum())
    per1k, per_correct = _cost_stats(total, n, n_correct)
    lat = matched[f"{prefix}_latency_ms"].dropna()
    return _point(workload, pop_sha, name, "every request", n=n, accuracy=n_correct / n if n else None, n_correct=n_correct, cost_per_1000_requests_usd=per1k, cost_complete=total is not None, cost_per_correct_decision_usd=per_correct,
                  latency_p50_ms=float(lat.quantile(0.5)) if len(lat) else None, latency_p95_ms=float(lat.quantile(0.95)) if len(lat) else None, latency_n=len(lat), latency_provenance=LATENCY_MEASURED, automation_coverage=1.0 if n else None)


def cascade_point(matched: pd.DataFrame, workload: dict[str, Any], pop_sha: str, threshold: float, architecture: str = "jev_to_llm", degenerate: str | None = None) -> dict[str, Any]:
    sim = simulate(matched, threshold)
    s = summarize_threshold(sim)
    n = s["n_total"]
    resolved = sim[sim["route"] != "unresolved_missing_llm"]
    n_correct = int(resolved["final_correct"].astype(bool).sum())
    total = s["total_cost_usd_estimated"]
    per1k, per_correct = _cost_stats(total, len(resolved), n_correct)
    return _point(workload, pop_sha, architecture, f"jev_threshold={threshold:.2f}", n=n, accuracy=s["final_accuracy"], n_correct=n_correct, cost_per_1000_requests_usd=per1k, cost_complete=total is not None, cost_per_correct_decision_usd=per_correct,
                  latency_p50_ms=s["path_latency_p50_ms"], latency_p95_ms=s["path_latency_p95_ms"], latency_n=n, latency_provenance=LATENCY_SIMULATED, automation_coverage=1.0 - s["n_unresolved_missing_llm"] / n if n else None,
                  degenerate_reason=degenerate, note=f"share handled by Jev {s['pct_handled_by_jev']:.3f}; fell back to LLM {s['pct_fell_back_to_llm']:.3f}")


def oracle_point(matched: pd.DataFrame, workload: dict[str, Any], pop_sha: str, jev_t: float, llm_t: float) -> dict[str, Any]:
    sim = simulate_full(matched, CascadeConfig(jev_threshold=jev_t, llm_threshold=llm_t), NoRules())
    s = summarize_full(sim)
    n = s["n_total"]
    correct = sim["final_correct"].astype(bool)
    return _point(workload, pop_sha, "rules_jev_llm_oracle", f"jev_threshold={jev_t:.2f}; llm_threshold={llm_t:.2f}", n=n, accuracy=s["overall_accuracy_including_oracle"], n_correct=int(correct.sum()), includes_oracle=True,
                  cost_per_1000_requests_usd=s["automated_stage_cost_per_1000_requests_usd_estimated"], cost_complete=False, cost_per_correct_decision_usd=None,
                  latency_p50_ms=s["path_latency_p50_ms"], latency_p95_ms=s["path_latency_p95_ms"], latency_n=s["path_latency_n"], latency_provenance=LATENCY_SIMULATED + "; automated-handled requests only, oracle latency NA",
                  automation_coverage=s["automation_coverage_excluding_oracle"], oracle_fallback_rate=s["coverage_oracle"],
                  note="accuracy includes ground-truth oracle answers by construction; cost covers automated stages only (oracle/rules cost NA), so it is NOT total system cost")


def workload_points(matched: pd.DataFrame, workload: dict[str, Any], jev_thresholds: list[float] = JEV_THRESHOLDS, llm_thresholds: list[float] = ORACLE_LLM_THRESHOLDS) -> list[dict[str, Any]]:
    sha = population_fingerprint(matched)
    points = [_point(workload, sha, "rules_only", "-", n=len(matched), status="not_available", note=NO_RULES_REASON),
              single_provider_point(matched, workload, sha, "jev", "jev_only"), single_provider_point(matched, workload, sha, "llm", "llm_only")]
    for t in jev_thresholds:
        points.append(cascade_point(matched, workload, sha, t))
        points.append(cascade_point(matched, workload, sha, t, "rules_jev_llm", "rules coverage is 0, so this is identical to jev_to_llm"))
    points += [oracle_point(matched, workload, sha, jt, lt) for jt in jev_thresholds for lt in llm_thresholds]
    assert_same_population(points)
    return points


def mark_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: among AUTOMATED points with complete cost in a workload, flag those no other point dominates."""
    out = df.copy()
    out["on_automated_cost_accuracy_frontier"] = False
    for _, idx in out.groupby("workload").groups.items():
        cand = out.loc[idx]
        cand = cand[(cand["status"] == "computed") & (~cand["includes_oracle"].astype(bool)) & cand["cost_complete"].astype(bool) & cand["accuracy"].notna() & (cand["degenerate_reason"].isna())]
        for i, a in cand.iterrows():
            dominated = ((cand["cost_per_1000_requests_usd"] <= a["cost_per_1000_requests_usd"]) & (cand["accuracy"] >= a["accuracy"]) & ((cand["cost_per_1000_requests_usd"] < a["cost_per_1000_requests_usd"]) | (cand["accuracy"] > a["accuracy"]))).any()
            out.loc[i, "on_automated_cost_accuracy_frontier"] = not dominated
    return out


def tradeoffs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Numeric pairwise contrasts, Jev-only vs LLM-only, per workload. Statements of difference, not a verdict."""
    out = []
    for w, g in df.groupby("workload"):
        j, l = g[g["architecture"] == "jev_only"].iloc[0], g[g["architecture"] == "llm_only"].iloc[0]
        out.append({"workload": w, "n": int(j["n"]), "accuracy_jev_only": j["accuracy"], "accuracy_llm_only": l["accuracy"], "accuracy_difference_jev_minus_llm": j["accuracy"] - l["accuracy"],
                    "cost_per_1000_jev_only_usd": j["cost_per_1000_requests_usd"], "cost_per_1000_llm_only_usd": l["cost_per_1000_requests_usd"],
                    "latency_p95_jev_only_ms": j["latency_p95_ms"], "latency_p95_llm_only_ms": l["latency_p95_ms"], "note": "descriptive contrast on the same requests; no architecture is declared best"})
    return out


def run_cost_quality(baseline_dir: Path = BASELINE_DIR, out_dir: Path = COST_QUALITY_DIR, verify: bool = True) -> dict[str, Any]:
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification: {errors}")
    existing = [str(out_dir / n) for n in OUTPUTS if (out_dir / n).exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing output(s): {existing}")
    source = baseline_dir / CANONICAL_NAME
    canonical = pd.read_parquet(source)
    matched, _ = build_matched(canonical)
    matched = attach_text(matched, canonical)
    points: list[dict[str, Any]] = []
    for (exp, k), grp in matched.groupby(["experiment", "num_choices"], sort=True):
        wl = {"workload": f"{exp}|k={int(k)}", "experiment": exp, "num_choices": int(k), "dataset": grp["dataset"].iloc[0]}
        points += workload_points(grp, wl)
    df = mark_frontier(pd.DataFrame(points, columns=COLUMNS))
    tidy_cols = ["workload", "architecture", "config", "n", "accuracy", "cost_per_1000_requests_usd", "cost_complete", "latency_p50_ms", "latency_p95_ms", "automation_coverage", "oracle_fallback_rate", "includes_oracle", "on_automated_cost_accuracy_frontier", "status"]
    summary = {
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "workloads": sorted(df["workload"].unique()),
        "n_points": len(df),
        "comparability": "every architecture point within a workload uses the same matched request set (population_sha256 identical); workloads are never mixed",
        "cost_provenance": COST_PROVENANCE,
        "latency_provenance": {"single_provider": LATENCY_MEASURED, "cascades": LATENCY_SIMULATED},
        "rules": {"available": False, "reason": NO_RULES_REASON},
        "oracle": "oracle architectures include ground-truth answers by construction; their cost/latency are partial and they are excluded from the frontier flag",
        "frontier_flag": "descriptive only; not a recommendation and not a winner",
        "tradeoffs_jev_only_vs_llm_only": tradeoffs(df),
        "no_best_architecture_selected": True,
        "interpretation": "not concluded here; choose an objective (cost, accuracy, latency, automation) before reading the frontier",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "architecture_frontier.csv", index=False)
    df[tidy_cols].to_csv(out_dir / "cost_accuracy_points.csv", index=False)
    (out_dir / "frontier_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

