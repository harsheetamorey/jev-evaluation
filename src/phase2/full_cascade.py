"""Phase II Step 15: the full decision cascade, as an OFFLINE simulation (no model calls).

    Request -> Rules -> can rules handle it?   YES -> rule output
                        NO -> Jev -> confidence >= jev_threshold?   YES -> Jev output
                                     NO -> LLM -> accepted/validated?   YES -> LLM output
                                                  NO -> HUMAN / ORACLE

*** The human is an ORACLE: a simulated stage that returns the ground-truth label. It is NOT a measured
human benchmark. Its accuracy is correct by construction, and its latency and cost are NOT modeled (NA).
No claim about real human accuracy, cost or speed should be drawn from these numbers. ***

Rules: this repo has NO rules baseline (`src/clients/rules_client.py` is an empty scaffold). Rule coverage is
therefore reported as it actually is, 0%, using `NoRules`, and nothing is invented to improve it. The rules
stage is a pluggable interface (`RulesHandler`) so real rules can be dropped in later and tested.

LLM validation: the LLM output is accepted iff it produced a prediction from the candidate set AND, when a
threshold is configured, its confidence >= `llm_threshold` (missing confidence cannot be validated -> oracle).

Thresholds are configuration, not conclusions: nothing here is a recommended operating point. Each request
records its stage reached, final handler, stage latencies/costs, Jev confidence and fallback reason.
Latency/cost are reported only where defensible: automated stages use recorded per-call values; rule and
oracle latency/cost are NA; a request that reaches the oracle has no total path latency.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.calibration import dedup_measurements
from phase2.cascade import KEY, LATENCY_PROVENANCE, COST_PROVENANCE, build_matched
from phase2.stress import PHASE2_RESULTS_DIR

FULL_CASCADE_DIR = PHASE2_RESULTS_DIR / "full_cascade"
HANDLERS = ("rules", "jev", "llm", "oracle")
OUTPUTS = ("full_cascade_per_request.csv", "full_cascade_grid.csv", "full_cascade_summary.json")
DEFAULT_JEV_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
DEFAULT_LLM_THRESHOLDS = [0.0, 0.5, 0.8]
ORACLE_NOTE = "oracle = ground-truth label (simulation), NOT a measured human; its accuracy is correct by construction and its latency/cost are NA"


class RulesHandler(Protocol):
    def handle(self, request: pd.Series) -> str | None:
        """Return a label if the rules can handle the request, else None."""


class NoRules:
    """The repository's actual rules baseline: none exists, so rules handle nothing."""

    def handle(self, request: pd.Series) -> str | None:
        return None


@dataclass(frozen=True)
class CascadeConfig:
    jev_threshold: float = 0.90
    llm_enabled: bool = True
    llm_threshold: float | None = 0.5
    oracle_enabled: bool = True
    rules_enabled: bool = True


def _num(v: Any) -> float:
    return float(v) if v is not None and not pd.isna(v) else float("nan")


def route_request(row: pd.Series, cfg: CascadeConfig, rules: RulesHandler) -> dict[str, Any]:
    """Walk one request through the stages. Returns the per-request record."""
    out: dict[str, Any] = {k: row.get(k) for k in (*KEY, "num_choices", "ground_truth")}
    out.update({"jev_confidence": _num(row.get("jev_confidence")), "jev_latency_ms": float("nan"), "llm_latency_ms": float("nan"), "jev_cost_usd": float("nan"), "llm_cost_usd": float("nan"),
                "rules_latency_ms": float("nan"), "oracle_latency_ms": float("nan"), "oracle_cost_usd": float("nan"), "rules_cost_usd": float("nan")})
    stages = ["rules"] if cfg.rules_enabled else []
    reasons: list[str] = []

    def finish(handler: str, prediction: Any, correct: Any) -> dict[str, Any]:
        out.update({"stage_reached": stages[-1] if stages else "none", "final_handler": handler, "final_prediction": prediction, "final_correct": correct, "fallback_reason": ";".join(reasons) or None, "stages_visited": "|".join(stages)})
        automated_lat = np.nansum([out["jev_latency_ms"], out["llm_latency_ms"]]) if any(s in stages for s in ("jev", "llm")) else float("nan")
        out["automated_stage_latency_ms"] = float(automated_lat)
        out["path_latency_ms"] = float(automated_lat) if handler in ("jev", "llm") else float("nan")  # oracle/unhandled paths have no defensible total
        out["automated_stage_cost_usd_estimated"] = float(np.nansum([out["jev_cost_usd"], out["llm_cost_usd"]])) if any(s in stages for s in ("jev", "llm")) else float("nan")
        return out

    if cfg.rules_enabled:
        label = rules.handle(row)
        if label is not None:
            return finish("rules", label, label == row.get("ground_truth"))
        reasons.append("rules_cannot_handle")

    stages.append("jev")
    out["jev_latency_ms"], out["jev_cost_usd"] = _num(row.get("jev_latency_ms")), _num(row.get("jev_estimated_cost_usd"))
    jev_pred = row.get("jev_prediction")
    if pd.notna(jev_pred) and round(_num(row.get("jev_confidence")), 6) >= cfg.jev_threshold:
        return finish("jev", jev_pred, bool(row.get("jev_correct")))
    reasons.append("jev_missing_prediction" if pd.isna(jev_pred) else "jev_below_threshold")

    if cfg.llm_enabled:
        stages.append("llm")
        out["llm_latency_ms"], out["llm_cost_usd"] = _num(row.get("llm_latency_ms")), _num(row.get("llm_estimated_cost_usd"))
        llm_pred, llm_conf = row.get("llm_prediction"), _num(row.get("llm_confidence"))
        if pd.isna(llm_pred):
            reasons.append("llm_missing_prediction")
        elif cfg.llm_threshold is not None and (np.isnan(llm_conf) or round(llm_conf, 6) < cfg.llm_threshold):
            reasons.append("llm_not_validated_confidence")
        else:
            return finish("llm", llm_pred, bool(row.get("llm_correct")))
    else:
        reasons.append("llm_disabled")

    if cfg.oracle_enabled:
        stages.append("oracle")
        return finish("oracle", row.get("ground_truth"), True)  # correct by construction: it IS the ground truth
    return finish("unhandled", None, None)


def simulate_full(matched: pd.DataFrame, cfg: CascadeConfig, rules: RulesHandler | None = None) -> pd.DataFrame:
    rules = rules or NoRules()
    return pd.DataFrame([route_request(r, cfg, rules) for _, r in matched.iterrows()])


def summarize_full(sim: pd.DataFrame) -> dict[str, Any]:
    n = len(sim)
    counts = {h: int((sim["final_handler"] == h).sum()) for h in (*HANDLERS, "unhandled")}
    automated = sim[sim["final_handler"].isin(["rules", "jev", "llm"])]
    handled = sim[sim["final_handler"] != "unhandled"]
    correct_all = handled["final_correct"].astype(bool)
    lat = sim["path_latency_ms"].dropna()
    cost = sim["automated_stage_cost_usd_estimated"]
    out = {
        "n_total": n,
        **{f"n_{h}": c for h, c in counts.items()},
        **{f"coverage_{h}": (c / n if n else None) for h, c in counts.items()},
        "automation_coverage_excluding_oracle": len(automated) / n if n else None,
        "overall_accuracy_including_oracle": float(correct_all.mean()) if len(handled) else None,
        "overall_accuracy_note": ORACLE_NOTE,
        "automated_accuracy_excluding_oracle": float(automated["final_correct"].astype(bool).mean()) if len(automated) else None,
        "stage_errors": {h: int((~sim[sim["final_handler"] == h]["final_correct"].astype(bool)).sum()) for h in ("rules", "jev", "llm")},
        "path_latency_n": len(lat),
        "path_latency_p50_ms": float(lat.quantile(0.5)) if len(lat) else None,
        "path_latency_p95_ms": float(lat.quantile(0.95)) if len(lat) else None,
        "path_latency_scope": "automated-handled requests only (rules/oracle latency not modeled); simulated sequential path",
        "automated_stage_cost_per_1000_requests_usd_estimated": float(cost.sum() / n * 1000) if n and cost.notna().sum() == n else None,
        "n_missing_automated_cost": int(cost.isna().sum() - (sim["stage_reached"].isin(["none", "rules"])).sum()),
        "cost_scope": "sum of recorded automated-stage estimates (Jev on every request that reaches it, LLM on fallbacks); oracle/rule cost NA, so this is NOT total system cost",
        "fallback_reasons": {k: int(v) for k, v in sim["fallback_reason"].dropna().str.split(";").explode().value_counts().items()},
    }
    return out


def attach_text(matched: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    """Add the request text (from the canonical baseline) so rules can inspect it."""
    txt = dedup_measurements(canonical.assign(candidates=canonical["candidates"].map(list)))
    txt = txt[txt["provider"] == "jev"][[*KEY, "text"]]
    return matched.merge(txt, on=KEY, how="left")


def grid(matched: pd.DataFrame, jev_thresholds: list[float], llm_thresholds: list[float | None], rules: RulesHandler | None = None) -> pd.DataFrame:
    rows = []
    for (exp, k), grp in matched.groupby(["experiment", "num_choices"], sort=True):
        for jt in jev_thresholds:
            for lt in llm_thresholds:
                cfg = CascadeConfig(jev_threshold=jt, llm_threshold=lt)
                s = summarize_full(simulate_full(grp, cfg, rules))
                rows.append({"experiment": exp, "num_choices": int(k), "jev_threshold": jt, "llm_threshold": lt, **{k2: (json.dumps(v) if isinstance(v, dict) else v) for k2, v in s.items() if k2 != "fallback_reasons"}, "fallback_reasons": json.dumps(s["fallback_reasons"])})
    return pd.DataFrame(rows)


def run_full_cascade(baseline_dir: Path = BASELINE_DIR, out_dir: Path = FULL_CASCADE_DIR, cfg: CascadeConfig = CascadeConfig(), jev_thresholds: list[float] = DEFAULT_JEV_THRESHOLDS, llm_thresholds: list[float | None] = DEFAULT_LLM_THRESHOLDS, verify: bool = True) -> dict[str, Any]:
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
    matched = attach_text(matched, canonical)
    rules = NoRules()
    per_request = simulate_full(matched, cfg, rules)
    table = grid(matched, jev_thresholds, llm_thresholds, rules)
    summary = {
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "n_requests": len(matched),
        "population": "requests with both a Jev and a reference-LLM prediction on the same example and contract (Step 14 matched set)",
        "per_request_config": asdict(cfg),
        "grid": {"jev_thresholds": jev_thresholds, "llm_thresholds": llm_thresholds},
        "rules": {"implementation": "NoRules", "reason": "the repository has no rules baseline (src/clients/rules_client.py is empty)", "actual_coverage": summarize_full(per_request)["coverage_rules"]},
        "overall": summarize_full(per_request),
        "assumptions": {"oracle": ORACLE_NOTE, "latency": LATENCY_PROVENANCE, "cost": COST_PROVENANCE},
        "thresholds_are_configuration_not_recommendations": True,
        "interpretation": "not concluded here",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    per_request.to_csv(out_dir / "full_cascade_per_request.csv", index=False)
    table.to_csv(out_dir / "full_cascade_grid.csv", index=False)
    (out_dir / "full_cascade_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
