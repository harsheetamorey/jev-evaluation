"""Phase II Step 2: does reported confidence correspond to actual correctness?

Reads the frozen canonical baseline (never the raw duplicated Phase I rows) and
writes tidy calibration tables. Makes no API calls and never modifies Phase I
or baseline files.

What "confidence" is here: `PredictionResult.confidence` is the probability
Jev/the LLM assigned to its CHOSEN label (`answer.probabilities.get(prediction)`),
stored to 2 decimals. The full distribution over candidates is NOT stored, and
this module never infers it.

Definitions
-----------
Bins: 10 equal-width bins [0.0,0.1), [0.1,0.2), ..., [0.9,1.0]. The last bin is
closed so confidence exactly 1.0 lands in it; confidence exactly 0.0 lands in the first.

Per bin b (n_b predictions): mean_confidence_b, accuracy_b = share correct,
calibration_gap_b = mean_confidence_b - accuracy_b (positive = overconfident).

ECE (Expected Calibration Error) = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|,
i.e. each bin weighted by its share of the N labeled predictions; empty bins contribute 0.

Brier score for correctness confidence = mean((confidence - correct)^2), correct in {0,1}.
This is a BINARY score on the chosen-label confidence, NOT a full multiclass Brier
score (which would need the whole probability distribution, which is not stored).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline

CALIBRATION_DIR = Path("data/results/phase2/calibration")
N_BINS = 10
BRIER_LABEL = "brier_correctness_binary (mean((confidence - correct)^2) on chosen-label confidence; NOT multiclass)"
ECE_FORMULA = "ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b| over 10 equal-width bins; last bin closed [0.9, 1.0]; empty bins contribute 0"
GROUP_COLUMNS = ["run_id", "experiment", "provider", "dataset", "num_choices"]

# level name -> (group columns, dedup repeated measurements first?, pooled across different workloads?)
LEVELS: dict[str, tuple[list[str], bool, bool]] = {
    "per_run": (["run_id", "experiment", "provider", "dataset", "num_choices"], False, False),
    "provider_dataset_experiment": (["provider", "dataset", "experiment", "num_choices"], True, False),
    "provider_dataset": (["provider", "dataset"], True, True),
    "provider": (["provider"], True, True),
    "all_pooled": ([], True, True),
}
# One logical measurement, used to collapse runs that re-measure the same example.
MEASUREMENT_KEY = ["experiment", "provider", "example_id", "locale"]


def prepare_predictions(canonical: pd.DataFrame) -> pd.DataFrame:
    """Add `num_choices`; return all canonical rows (eligibility is applied later)."""
    df = canonical.copy()
    df["num_choices"] = df["candidates"].map(len)
    return df


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Labeled predictions: both `correct` and `confidence` present (errored calls have neither)."""
    out = df[df["correct"].notna() & df["confidence"].notna()].copy()
    out["correct"] = out["correct"].astype(bool)
    if ((out["confidence"] < 0) | (out["confidence"] > 1)).any():
        raise ValueError("confidence values outside [0, 1] found")
    return out


def dedup_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row per logical measurement, as in the Step 1 baseline rule (outcome-independent)."""
    df = df.sort_values("raw_row")
    return df.drop_duplicates(subset=MEASUREMENT_KEY, keep="last")


def bin_index(confidence: pd.Series) -> pd.Series:
    """Bin 0..9. round() guards float noise (e.g. 0.3*10); 1.0 goes to the closed last bin."""
    return np.minimum(np.floor(np.round(confidence * N_BINS, 9)), N_BINS - 1).astype(int)


def reliability_bins(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bin (all 10, including empty ones): n, mean confidence, accuracy, gap."""
    idx = bin_index(df["confidence"]) if len(df) else pd.Series([], dtype=int)
    rows = []
    for b in range(N_BINS):
        sub = df[idx == b] if len(df) else df
        n = len(sub)
        mean_conf = float(sub["confidence"].mean()) if n else np.nan
        acc = float(sub["correct"].astype(float).mean()) if n else np.nan
        rows.append(
            {
                "bin_index": b,
                "bin_low": b / N_BINS,
                "bin_high": (b + 1) / N_BINS,
                "bin_label": f"[{b / N_BINS:.1f}, {(b + 1) / N_BINS:.1f}{']' if b == N_BINS - 1 else ')'}",
                "sample_count": n,
                "mean_confidence": mean_conf,
                "actual_accuracy": acc,
                "calibration_gap": mean_conf - acc if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame) -> float | None:
    total = bins["sample_count"].sum()
    if total == 0:
        return None
    populated = bins[bins["sample_count"] > 0]
    return float((populated["sample_count"] / total * (populated["actual_accuracy"] - populated["mean_confidence"]).abs()).sum())


def brier_correctness(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(((df["confidence"] - df["correct"].astype(float)) ** 2).mean())


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    correct = df[df["correct"].astype(bool)]
    incorrect = df[~df["correct"].astype(bool)]
    bins = reliability_bins(df)
    return {
        "n": len(df),
        "n_distinct_examples": int(df["example_id"].nunique()),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "accuracy": float(df["correct"].astype(float).mean()) if len(df) else None,
        "mean_confidence": float(df["confidence"].mean()) if len(df) else None,
        "mean_confidence_correct": float(correct["confidence"].mean()) if len(correct) else None,
        "mean_confidence_incorrect": float(incorrect["confidence"].mean()) if len(incorrect) else None,
        "ece": expected_calibration_error(bins),
        "brier_correctness": brier_correctness(df),
        "n_populated_bins": int((bins["sample_count"] > 0).sum()),
        "n_bins_under_30": int(((bins["sample_count"] > 0) & (bins["sample_count"] < 30)).sum()),
    }


def calibration_tables(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the tidy (bins, summary) tables across every analysis level."""
    all_rows = prepare_predictions(canonical)
    bin_frames, summary_rows = [], []
    for level, (group_cols, dedup, pooled) in LEVELS.items():
        base = eligible(dedup_measurements(all_rows) if dedup else all_rows)
        groups = base.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), base)]
        for key, grp in groups:
            key = key if isinstance(key, tuple) else (key,)
            ids = {col: val for col, val in zip(group_cols, key, strict=True)}
            meta = {"level": level, "pooled_across_workloads": pooled, **{c: ids.get(c) for c in GROUP_COLUMNS}}
            bin_frames.append(pd.concat([pd.DataFrame([meta] * N_BINS), reliability_bins(grp)], axis=1))
            summary_rows.append({**meta, **summarize(grp), "brier_form": BRIER_LABEL})
    return pd.concat(bin_frames, ignore_index=True), pd.DataFrame(summary_rows)


def run_calibration(
    baseline_dir: Path = BASELINE_DIR,
    out_dir: Path = CALIBRATION_DIR,
    verify: bool = True,
) -> dict[str, Any]:
    """Verify the frozen baseline, compute calibration from its canonical view, write tidy tables."""
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification, not computing calibration: {errors}")
    source = baseline_dir / CANONICAL_NAME
    outputs = {name: out_dir / name for name in ("calibration_bins.csv", "calibration_summary.csv", "calibration_meta.json")}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing calibration output(s): {existing}")

    canonical = pd.read_parquet(source)
    bins, summary = calibration_tables(canonical)
    eligible_rows = eligible(prepare_predictions(canonical))
    meta = {
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "canonical_rows": len(canonical),
        "eligible_rows": len(eligible_rows),
        "excluded_rows_no_label_or_confidence": len(canonical) - len(eligible_rows),
        "ece_formula": ECE_FORMULA,
        "brier_form": BRIER_LABEL,
        "confidence_representation": "probability of the chosen label only, stored to 2 decimals; full class distribution not stored",
        "bins": [f"[{b / N_BINS:.1f}, {(b + 1) / N_BINS:.1f}{']' if b == N_BINS - 1 else ')'}" for b in range(N_BINS)],
        "levels": {name: {"group_columns": cols, "dedup_repeated_measurements": dedup, "pooled_across_workloads": pooled} for name, (cols, dedup, pooled) in LEVELS.items()},
        "dedup_key_for_pooled_levels": MEASUREMENT_KEY,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    bins.to_csv(outputs["calibration_bins.csv"], index=False)
    summary.to_csv(outputs["calibration_summary.csv"], index=False)
    outputs["calibration_meta.json"].write_text(json.dumps(meta, indent=2))
    return meta
