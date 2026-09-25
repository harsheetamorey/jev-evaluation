"""Phase II Step 3: selective prediction / risk-coverage.

Answers: if we reject uncertain predictions, how much does accuracy improve and how
much automation do we lose? Reads the frozen canonical baseline, makes no API calls,
and never modifies Phase I or baseline files.

Rule, per labeled prediction: confidence >= threshold -> ACCEPT, otherwise FALLBACK/ABSTAIN.

Definitions (per group and threshold):
    coverage          = accepted / total
    rejected_fraction = rejected / total = 1 - coverage
    accepted_accuracy = accepted_correct / accepted     (None when accepted == 0)
    risk              = accepted_incorrect / accepted   (None when accepted == 0)

Nothing here assumes accuracy rises with the threshold, and no "best" threshold is chosen:
this step describes the trade-off; operating points are picked later, when building cascades.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.calibration import dedup_measurements, eligible, prepare_predictions

SELECTIVE_DIR = Path("data/results/phase2/selective_prediction")
THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(50)]  # 0.50 ... 0.99
STANDARD_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
GROUP_COLUMNS = ["run_id", "experiment", "provider", "dataset", "num_choices", "locale"]

# level -> (group columns, dedup repeated measurements first?, pooled across different workloads?)
LEVELS: dict[str, tuple[list[str], bool, bool]] = {
    "per_run": (["run_id", "experiment", "provider", "dataset", "num_choices"], False, False),
    "per_run_locale": (["run_id", "experiment", "provider", "dataset", "num_choices", "locale"], False, False),
    "provider_dataset_experiment": (["provider", "dataset", "experiment", "num_choices"], True, False),
    "provider_dataset_experiment_locale": (["provider", "dataset", "experiment", "num_choices", "locale"], True, False),
    "provider_dataset": (["provider", "dataset"], True, True),
    "provider": (["provider"], True, True),
    "all_pooled": ([], True, True),
}
LOCALE_LEVELS = {"per_run_locale", "provider_dataset_experiment_locale"}


def risk_coverage(df: pd.DataFrame, thresholds: list[float] = THRESHOLDS) -> pd.DataFrame:
    """One row per threshold. `df` must hold labeled rows (`confidence`, boolean `correct`).

    Accepted accuracy and risk are None (not 0, not inferred) when nothing is accepted.
    """
    total = len(df)
    conf = df["confidence"].round(6)
    correct = df["correct"].astype(bool)
    rows = []
    for t in thresholds:
        accepted_mask = conf >= t
        accepted = int(accepted_mask.sum())
        accepted_correct = int((accepted_mask & correct).sum())
        rejected = total - accepted
        rejected_correct = int((~accepted_mask & correct).sum())
        rows.append(
            {
                "threshold": t,
                "is_standard_threshold": t in STANDARD_THRESHOLDS,
                "total": total,
                "accepted": accepted,
                "rejected": rejected,
                "coverage": accepted / total if total else None,
                "rejected_fraction": rejected / total if total else None,
                "accepted_correct": accepted_correct,
                "accepted_incorrect": accepted - accepted_correct,
                "accepted_accuracy": accepted_correct / accepted if accepted else None,
                "risk": (accepted - accepted_correct) / accepted if accepted else None,
                "rejected_correct": rejected_correct,
                "rejected_accuracy": rejected_correct / rejected if rejected else None,
            }
        )
    return pd.DataFrame(rows)


def coverage_is_non_increasing(curve: pd.DataFrame) -> bool:
    values = curve["coverage"].dropna().tolist()
    return all(a >= b for a, b in zip(values, values[1:], strict=False))


def accuracy_is_non_decreasing(curve: pd.DataFrame) -> bool:
    """Whether accepted accuracy never fell as the threshold rose. Allowed to be False."""
    values = curve["accepted_accuracy"].dropna().tolist()
    return all(a <= b for a, b in zip(values, values[1:], strict=False))


def risk_coverage_tables(canonical: pd.DataFrame, thresholds: list[float] = THRESHOLDS) -> pd.DataFrame:
    """Tidy risk-coverage rows for every analysis level and group."""
    all_rows = prepare_predictions(canonical)
    frames = []
    for level, (group_cols, dedup, pooled) in LEVELS.items():
        rows = dedup_measurements(all_rows) if dedup else all_rows
        base = eligible(rows)
        if level in LOCALE_LEVELS:
            base = base[base["locale"].notna()]
        groups = base.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), base)]
        for key, grp in groups:
            key = key if isinstance(key, tuple) else (key,)
            ids = dict(zip(group_cols, key, strict=True))
            meta = {"level": level, "pooled_across_workloads": pooled, **{c: ids.get(c) for c in GROUP_COLUMNS}}
            curve = risk_coverage(grp, thresholds)
            frames.append(pd.concat([pd.DataFrame([meta] * len(curve)), curve], axis=1))
    return pd.concat(frames, ignore_index=True)


def summarize_tables(table: pd.DataFrame) -> dict[str, Any]:
    """Compact per-group facts at the standard thresholds, plus monotonicity observations."""
    groups = []
    for _, curve in table.groupby(["level", *GROUP_COLUMNS], sort=True, dropna=False):
        first = curve.iloc[0]
        standard = curve[curve["is_standard_threshold"]]
        groups.append(
            {
                "level": first["level"],
                "pooled_across_workloads": bool(first["pooled_across_workloads"]),
                **{c: (None if pd.isna(first[c]) else first[c]) for c in GROUP_COLUMNS},
                "total": int(first["total"]),
                "coverage_non_increasing": coverage_is_non_increasing(curve),
                "accepted_accuracy_non_decreasing": accuracy_is_non_decreasing(curve),
                "thresholds_with_zero_accepted": [float(t) for t in curve.loc[curve["accepted"] == 0, "threshold"]],
                "at_standard_thresholds": [
                    {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
                    for row in standard[["threshold", "accepted", "rejected", "coverage", "accepted_accuracy", "risk"]].to_dict("records")
                ],
            }
        )
    return {"n_groups": len(groups), "groups": groups}


def run_selective(baseline_dir: Path = BASELINE_DIR, out_dir: Path = SELECTIVE_DIR, verify: bool = True) -> dict[str, Any]:
    """Verify the frozen baseline, compute risk-coverage tables from its canonical view, write them."""
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification, not computing risk-coverage: {errors}")
    source = baseline_dir / CANONICAL_NAME
    csv_path, json_path = out_dir / "risk_coverage.csv", out_dir / "risk_coverage_summary.json"
    existing = [str(p) for p in (csv_path, json_path) if p.exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing selective-prediction output(s): {existing}")

    canonical = pd.read_parquet(source)
    table = risk_coverage_tables(canonical)
    labeled = eligible(prepare_predictions(canonical))
    summary = {
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "canonical_rows": len(canonical),
        "labeled_rows_analyzed": len(labeled),
        "excluded_rows_no_label_or_confidence": len(canonical) - len(labeled),
        "rule": "ACCEPT if confidence >= threshold, else FALLBACK/ABSTAIN",
        "definitions": {
            "coverage": "accepted / total",
            "risk": "accepted_incorrect / accepted (null when accepted == 0)",
            "accepted_accuracy": "accepted_correct / accepted (null when accepted == 0)",
        },
        "thresholds": {"sweep": [THRESHOLDS[0], THRESHOLDS[-1], 0.01], "standard": STANDARD_THRESHOLDS},
        "pooled_levels_dedup_repeated_measurements": True,
        "no_best_threshold_selected": True,
        **summarize_tables(table),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary
