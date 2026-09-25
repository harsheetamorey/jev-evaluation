"""Phase II Step 12: multilingual reliability deep dive (zero API calls).

Are failures language-specific, intent-specific, or consistently difficult across languages?

Works only from the frozen canonical baseline's MASSIVE multilingual predictions. Whatever aligned
population is truly present is used and its size reported (it is 100 aligned IDs x 8 locales, not
the 250 the Phase I plan mentioned). Repeated measurements of the same (provider, example, locale)
are collapsed with the same latest-row rule used elsewhere in Phase II; a pure `aligned_records`
call REJECTS duplicate locale rows rather than silently averaging them.

Per aligned source id and provider we build ONE record with correctness/confidence for every
available locale, then classify it deterministically from the correctness pattern:

    all_correct                 right in every available locale
    language_localized_failure  wrong in 1-2 locales
    mixed_unstable              wrong in 3+ locales but fewer than the global-difficulty share
    globally_difficult          wrong in >= GLOBAL_WRONG_SHARE of the available locales (all-wrong included)
    insufficient_locales        fewer than MIN_LOCALES locales available (e.g. errored calls)

`prediction_unstable` separately flags examples where the predicted label differs across locales.
Nothing here ranks languages or intents; every rate is reported with its n.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_loaders.massive import LOCALES
from evaluation.metrics import wilson_ci
from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.calibration import dedup_measurements

MULTILINGUAL_DIR = Path("data/results/phase2/multilingual")
EXPERIMENT = "multilingual"
MIN_LOCALES = 6
LOCALIZED_MAX_WRONG = 2
GLOBAL_WRONG_SHARE = 0.75
MIN_CELL_N = 5  # intent x language accuracy is only reported when at least this many examples exist
CLASSES = ("all_correct", "language_localized_failure", "mixed_unstable", "globally_difficult", "insufficient_locales")
OUTPUTS = ("aligned_reliability.csv", "intent_reliability.csv", "intent_language_reliability.csv", "language_reliability.csv", "consistency_summary.json")


def classify(n_available: int, n_wrong: int) -> str:
    """Deterministic class from correctness counts alone."""
    if n_available < MIN_LOCALES:
        return "insufficient_locales"
    if n_wrong == 0:
        return "all_correct"
    if n_wrong / n_available >= GLOBAL_WRONG_SHARE:
        return "globally_difficult"
    if n_wrong <= LOCALIZED_MAX_WRONG:
        return "language_localized_failure"
    return "mixed_unstable"


def aligned_records(df: pd.DataFrame, locales: list[str] = LOCALES) -> pd.DataFrame:
    """One row per (provider, source_example_id). `df` must hold at most one row per (provider, example_id, locale)."""
    dup = df.duplicated(["provider", "example_id", "locale"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (provider, example_id, locale) rows; deduplicate before aligning")
    unknown = set(df["locale"]) - set(locales)
    if unknown:
        raise ValueError(f"unexpected locales: {sorted(unknown)}")
    rows = []
    for (provider, source), grp in df.groupby(["provider", "example_id"], sort=True):
        if grp["ground_truth"].nunique() != 1:
            raise ValueError(f"{source}: ground truth differs across locales; the example is not aligned")
        scored = grp[grp["correct"].notna()]
        by_locale = scored.set_index("locale")
        n_avail, n_correct = len(scored), int(scored["correct"].astype(bool).sum())
        preds = grp[grp["prediction"].notna()]
        rec: dict[str, Any] = {
            "provider": provider,
            "source_example_id": source,
            "intent": grp["ground_truth"].iloc[0],
            "n_locales_available": n_avail,
            "n_correct": n_correct,
            "n_wrong": n_avail - n_correct,
            "all_locales_correct": n_avail == len(locales) and n_correct == n_avail,
            "all_locales_wrong": n_avail == len(locales) and n_correct == 0,
            "classification": classify(n_avail, n_avail - n_correct),
            "n_distinct_predictions": int(preds["prediction"].nunique()),
            "prediction_unstable": bool(preds["prediction"].nunique() > 1),
            "mean_confidence": float(scored["confidence"].mean()) if n_avail else None,
            "min_confidence": float(scored["confidence"].min()) if n_avail else None,
        }
        for loc in locales:
            rec[f"correct_{loc}"] = bool(by_locale.loc[loc, "correct"]) if loc in by_locale.index else None
        rows.append(rec)
    return pd.DataFrame(rows)


def language_reliability(df: pd.DataFrame, aligned: pd.DataFrame, locales: list[str] = LOCALES) -> pd.DataFrame:
    rows = []
    for (provider, locale), grp in df.groupby(["provider", "locale"], sort=True):
        scored = grp[grp["correct"].notna()]
        correct = int(scored["correct"].astype(bool).sum())
        lo, hi = wilson_ci(correct, len(scored))
        prov = aligned[aligned["provider"] == provider]
        col = f"correct_{locale}"
        wrong_here = prov[prov[col].eq(False)]
        rows.append(
            {
                "provider": provider,
                "locale": locale,
                "n_rows": len(grp),
                "n_scored": len(scored),
                "n_errors_excluded": len(grp) - len(scored),
                "accuracy": correct / len(scored) if len(scored) else None,
                "accuracy_ci95_low": lo,
                "accuracy_ci95_high": hi,
                "mean_confidence": float(scored["confidence"].mean()) if len(scored) else None,
                "n_examples_wrong_here": len(wrong_here),
                "n_wrong_here_only": int((wrong_here["n_wrong"] == 1).sum()),
                "n_wrong_here_and_in_most_languages": int((wrong_here["classification"] == "globally_difficult").sum()),
            }
        )
    return pd.DataFrame(rows)


def intent_reliability(df: pd.DataFrame, aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, by_lang = [], []
    for (provider, intent), grp in df.groupby(["provider", "ground_truth"], sort=True):
        scored = grp[grp["correct"].notna()]
        a = aligned[(aligned["provider"] == provider) & (aligned["intent"] == intent)]
        overall.append(
            {
                "provider": provider,
                "intent": intent,
                "n_aligned_examples": len(a),
                "n_rows": len(grp),
                "n_scored": len(scored),
                "accuracy": float(scored["correct"].astype(bool).mean()) if len(scored) else None,
                "n_examples_failing_in_multiple_languages": int((a["n_wrong"] >= 2).sum()),
                "n_examples_globally_difficult": int((a["classification"] == "globally_difficult").sum()),
                "n_examples_language_localized": int((a["classification"] == "language_localized_failure").sum()),
                "note": "small n: do not rank" if len(a) < MIN_CELL_N else "",
            }
        )
        for locale, cell in scored.groupby("locale", sort=True):
            n = len(cell)
            by_lang.append({"provider": provider, "intent": intent, "locale": locale, "n": n, "accuracy": float(cell["correct"].astype(bool).mean()) if n >= MIN_CELL_N else None, "accuracy_reported": n >= MIN_CELL_N, "n_wrong": int((~cell["correct"].astype(bool)).sum())})
    return pd.DataFrame(overall), pd.DataFrame(by_lang)


def consistency_summary(aligned: pd.DataFrame, locales: list[str] = LOCALES) -> dict[str, Any]:
    out: dict[str, Any] = {"expected_locales": locales, "thresholds": {"min_locales": MIN_LOCALES, "localized_max_wrong": LOCALIZED_MAX_WRONG, "global_wrong_share": GLOBAL_WRONG_SHARE, "min_cell_n": MIN_CELL_N}, "providers": {}}
    for provider, grp in aligned.groupby("provider", sort=True):
        full = grp[grp["n_locales_available"] == len(locales)]
        out["providers"][provider] = {
            "n_aligned_examples": len(grp),
            "n_with_all_locales_scored": len(full),
            "all_8_correct": int(grp["all_locales_correct"].sum()),
            "all_8_wrong": int(grp["all_locales_wrong"].sum()),
            "n_correct_distribution": {str(k): int(v) for k, v in full["n_correct"].value_counts().sort_index().items()},
            "classification_counts": {c: int((grp["classification"] == c).sum()) for c in CLASSES},
            "n_prediction_unstable": int(grp["prediction_unstable"].sum()),
            "mean_locales_correct_when_full": float(full["n_correct"].mean()) if len(full) else None,
        }
    return out


def analyze_multilingual(canonical: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw = canonical[canonical["experiment"] == EXPERIMENT].copy()
    if raw.empty:
        raise BaselineError("no multilingual rows in the canonical baseline")
    raw["candidates"] = raw["candidates"].map(list)
    deduped = dedup_measurements(raw.assign(example_id=raw["example_id"]))
    aligned = aligned_records(deduped)
    lang = language_reliability(deduped, aligned)
    intent, intent_lang = intent_reliability(deduped, aligned)
    summary = consistency_summary(aligned)
    summary["population"] = {
        "canonical_multilingual_rows": len(raw),
        "rows_after_collapsing_repeated_measurements": len(deduped),
        "repeated_measurement_rows_removed": len(raw) - len(deduped),
        "n_aligned_source_ids": int(deduped["example_id"].nunique()),
        "n_intents": int(deduped["ground_truth"].nunique()),
        "locales_found": sorted(deduped["locale"].unique()),
        "rows_without_a_score_excluded": int(deduped["correct"].isna().sum()),
        "note": "aligned-ID count is whatever the canonical baseline holds; it is not assumed to be 250",
    }
    summary["interpretation"] = "not concluded here; every rate is shown with n and no language or intent is ranked"
    return {"aligned_reliability": aligned, "language_reliability": lang, "intent_reliability": intent, "intent_language_reliability": intent_lang}, summary


def run_multilingual(baseline_dir: Path = BASELINE_DIR, out_dir: Path = MULTILINGUAL_DIR, verify: bool = True) -> dict[str, Any]:
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification: {errors}")
    existing = [str(out_dir / n) for n in OUTPUTS if (out_dir / n).exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing output(s): {existing}")
    source = baseline_dir / CANONICAL_NAME
    tables, summary = analyze_multilingual(pd.read_parquet(source))
    summary["source_file"], summary["source_sha256"] = str(source), sha256_file(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "consistency_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
