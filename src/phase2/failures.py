"""Phase II Step 13: failure taxonomy and Failure Museum DATA MODEL (no UI, no model calls).

Consolidates failures recorded by earlier Phase II work into ONE normalized dataset:
  - the frozen canonical baseline (every wrong labeled prediction, plus multilingual language patterns)
  - each stress experiment's recorded results, if they exist yet (ambiguity, stability, noise,
    context pollution, context relevance, OOD, adversarial, choice overlap)

Experiments with no recorded results are reported as missing, never fabricated.

Tagging rules (`failure_tags` is a LIST; a failure can carry several tags, and no single cause is forced):
  * a CAUSE tag is added only when the experiment CONDITION itself establishes it, e.g. the adversarial
    experiment -> adversarial_manipulation, an OVERLAPPING choice set -> semantic_overlap, an ambiguity
    level other than CLEAR -> ambiguous_intent. We never infer a cause because two labels merely sound similar.
  * confidence tags use explicit, configurable thresholds (defaults below). They are NOT presented as
    scientifically optimal, just as a way to slice failures:
        high_confidence_wrong : confidence >= HIGH_CONFIDENCE_THRESHOLD
        low_confidence_wrong  : confidence <= LOW_CONFIDENCE_THRESHOLD
  * `primary_failure_type` is set only when directly determined by the experiment condition; else null.
  * a failure with no tag at all is "unclassified": the evidence does not establish a cause.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from phase2.baseline import BASELINE_DIR, CANONICAL_NAME, MANIFEST_NAME, BaselineError, sha256_file, verify_baseline
from phase2.calibration import dedup_measurements
from phase2.multilingual import aligned_records
from phase2.stress import PHASE2_RESULTS_DIR

FAILURES_DIR = PHASE2_RESULTS_DIR / "failures"
HIGH_CONFIDENCE_THRESHOLD = 0.90
LOW_CONFIDENCE_THRESHOLD = 0.60
FAILURE_TAGS = ("ambiguous_intent", "semantic_overlap", "insufficient_context", "irrelevant_context", "ood_input", "language_specific", "surface_noise", "adversarial_manipulation", "high_confidence_wrong", "low_confidence_wrong")
REQUIRED_FIELDS = ("failure_id", "source_example_id", "experiment", "provider", "prediction", "failure_tags", "source_file", "source_key")
COLUMNS = ["failure_id", "source_example_id", "experiment", "dataset", "provider", "input", "transformed_input", "ground_truth", "prediction", "confidence", "failure_tags", "primary_failure_type", "stress_condition", "locale", "metadata", "source_file", "source_key", "run_id"]
OUTPUT_FILES = ("failures.jsonl", "failures.csv", "failures_summary.json")


def make_failure_id(experiment: str, provider: str, source_file: str, source_key: str) -> str:
    """Deterministic id: the same failure always gets the same id, whatever the run order."""
    digest = hashlib.sha256("|".join([experiment, provider, source_file, source_key]).encode("utf-8")).hexdigest()
    return f"F-{digest[:16]}"


def confidence_tags(confidence: float | None, high: float = HIGH_CONFIDENCE_THRESHOLD, low: float = LOW_CONFIDENCE_THRESHOLD) -> list[str]:
    if confidence is None or pd.isna(confidence):
        return []
    c = round(float(confidence), 6)
    return (["high_confidence_wrong"] if c >= high else []) + (["low_confidence_wrong"] if c <= low else [])


def _clean(v: Any) -> Any:
    return None if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else v


def failure_record(*, experiment: str, dataset: str, provider: str, source_example_id: str, source_file: str, source_key: str, input_text: str | None, transformed_input: str | None,
                   ground_truth: Any, prediction: Any, confidence: Any, cause_tags: list[str], primary: str | None, condition: str | None, locale: Any, metadata: dict[str, Any],
                   run_id: Any = None, high: float = HIGH_CONFIDENCE_THRESHOLD, low: float = LOW_CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    conf = _clean(confidence)
    tags = list(dict.fromkeys([*cause_tags, *confidence_tags(conf, high, low)]))
    return {
        "failure_id": make_failure_id(experiment, provider, source_file, source_key),
        "source_example_id": str(source_example_id),
        "experiment": experiment,
        "dataset": dataset,
        "provider": provider,
        "input": input_text,
        "transformed_input": transformed_input,
        "ground_truth": _clean(ground_truth),
        "prediction": _clean(prediction),
        "confidence": None if conf is None else float(conf),
        "failure_tags": tags,
        "primary_failure_type": primary,
        "stress_condition": condition,
        "locale": _clean(locale),
        "metadata": {k: _clean(v) for k, v in metadata.items()},
        "source_file": source_file,
        "source_key": source_key,
        "run_id": _clean(run_id),
    }


# --- baseline (canonical) ------------------------------------------------------------------------


def baseline_failures(canonical: pd.DataFrame, source_file: str, high: float = HIGH_CONFIDENCE_THRESHOLD, low: float = LOW_CONFIDENCE_THRESHOLD) -> list[dict[str, Any]]:
    """Every wrong labeled prediction in the canonical baseline (repeated measurements collapsed to the latest row).

    Multilingual failures get `language_specific` ONLY when that example is wrong in just 1-2 of its 8 locales
    (Step 12's `language_localized_failure`); failures shared across most languages are not attributed to language.
    """
    df = dedup_measurements(canonical.assign(candidates=canonical["candidates"].map(list)))
    localized: set[tuple[str, str]] = set()
    ml = df[df["experiment"] == "multilingual"]
    if len(ml):
        aligned = aligned_records(ml)
        for _, a in aligned[aligned["classification"] == "language_localized_failure"].iterrows():
            localized |= {(a["provider"], f"{a['source_example_id']}:{loc}") for loc in [c[8:] for c in a.index if c.startswith("correct_") and pd.notna(a[c]) and not a[c]]}
    out = []
    for _, r in df[df["correct"].eq(False)].iterrows():
        loc_key = (r["provider"], f"{r['example_id']}:{r['locale']}")
        is_localized = r["experiment"] == "multilingual" and loc_key in localized
        out.append(
            failure_record(
                experiment=r["experiment"], dataset=r["dataset"], provider=r["provider"], source_example_id=r["example_id"], source_file=source_file,
                source_key=f"{r['example_id']}:{r['locale']}" if pd.notna(r["locale"]) else str(r["example_id"]),
                input_text=r["text"], transformed_input=None, ground_truth=r["ground_truth"], prediction=r["prediction"], confidence=r["confidence"],
                cause_tags=["language_specific"] if is_localized else [], primary=None, condition=None, locale=r["locale"],
                metadata={"n_choices": len(r["candidates"]), "canonical_raw_row": int(r["raw_row"]), "candidates": list(r["candidates"])}, run_id=r["run_id"], high=high, low=low,
            )
        )
    return out


# --- stress experiments --------------------------------------------------------------------------


def with_base(df: pd.DataFrame, is_base: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """Attach each row's own base (clean/original) prediction and correctness, by (provider, source_example_id)."""
    base = df[is_base(df)][["provider", "source_example_id", "prediction", "correct"]].rename(columns={"prediction": "base_prediction", "correct": "base_correct"})
    return df.merge(base.drop_duplicates(["provider", "source_example_id"]), on=["provider", "source_example_id"], how="left")


def _adapter(experiment: str, condition_col: str, is_failure: Callable[[pd.Series], bool], causes: Callable[[pd.Series], tuple[list[str], str | None]], input_col: str, meta_cols: list[str],
             base_mask: Callable[[pd.DataFrame], pd.Series] | None = None, prepare: Callable[[pd.DataFrame], pd.DataFrame] | None = None):
    def run(df: pd.DataFrame, source_file: str, high: float, low: float) -> list[dict[str, Any]]:
        d = prepare(df) if prepare else df
        d = with_base(d, base_mask) if base_mask else d
        out = []
        for _, r in d[d["prediction"].notna()].iterrows():
            if not is_failure(r):
                continue
            tags, primary = causes(r)
            text_in = r[input_col] if input_col in r and pd.notna(r[input_col]) else r.get("text")
            out.append(
                failure_record(
                    experiment=experiment, dataset=str(r.get("dataset", "stress")), provider=r["provider"], source_example_id=r["source_example_id"], source_file=source_file, source_key=str(r["variant_id"]),
                    input_text=text_in, transformed_input=r["text"] if r["text"] != text_in else None, ground_truth=r.get("ground_truth"), prediction=r["prediction"], confidence=r["confidence"],
                    cause_tags=tags, primary=primary, condition=str(r[condition_col]) if condition_col in r else None, locale=r.get("locale"),
                    metadata={c: (r[c] if c in r else None) for c in meta_cols}, run_id=r.get("run_id"), high=high, low=low,
                )
            )
        return out

    return run


def _wrong(r: pd.Series) -> bool:
    """A scored, incorrect prediction (unscored rows, where `correct` is null, are not failures)."""
    c = r.get("correct")
    return c is not None and bool(pd.notna(c)) and not bool(c)


def _ambiguity_prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    acc = [json.loads(a) if isinstance(a, str) else list(a) for a in d["acceptable_intents"]]
    d["acceptable_ok"] = [p in a for p, a in zip(d["prediction"], acc, strict=True)]
    d["input_shown"] = d["original_text"]
    return d


ADAPTERS: dict[str, tuple[str, Callable]] = {
    "ambiguity": ("ambiguity", _adapter("ambiguity", "ambiguity_level", lambda r: not r["acceptable_ok"], lambda r: (["ambiguous_intent"], "ambiguous_intent") if r["ambiguity_level"] != "CLEAR" else ([], None), "original_text", ["ambiguity_level", "acceptable_intents", "primary_expected_intent", "family"], base_mask=lambda d: d["ambiguity_level"] == "CLEAR", prepare=_ambiguity_prepare)),
    "stability": ("stability", _adapter("stability", "variant_type", _wrong, lambda r: (["surface_noise"], "surface_noise") if r["variant_family"] == "surface" else ([], None), "original_text", ["variant_type", "variant_family", "intent"], base_mask=lambda d: d["variant_type"] == "original")),
    "noise": ("noise", _adapter("noise", "noise_type", lambda r: _wrong(r) and r["severity"] > 0, lambda r: (["surface_noise"], "surface_noise" if r["semantic_integrity"] == "valid" else None), "original_text", ["noise_type", "severity", "semantic_integrity", "intent"], base_mask=lambda d: d["severity"] == 0)),
    "context_pollution": ("context_pollution", _adapter("context_pollution", "context_level", lambda r: _wrong(r) and r["context_level"] != "none", lambda r: (["irrelevant_context"], "irrelevant_context"), "base_message", ["context_level", "n_history_items", "history_chars", "context_source_ids"], base_mask=lambda d: d["context_level"] == "none")),
    "context_relevance": ("context", _adapter("context_relevance", "context_condition", _wrong, lambda r: {"irrelevant_context": (["irrelevant_context"], "irrelevant_context"), "no_context": (["insufficient_context"], "insufficient_context")}.get(r["context_condition"], ([], None)), "base_message", ["context_condition", "relevance_cues", "intent"], base_mask=lambda d: d["context_condition"] == "no_context")),
    "ood": ("ood", _adapter("ood", "contract_type", lambda r: (r["population"] == "ood" and r["prediction"] != "other") or (r["population"] == "in_domain" and _wrong(r)), lambda r: (["ood_input"], "ood_input") if r["population"] == "ood" else ([], None), "text", ["population", "contract_type", "massive_intent", "matched_control_id"])),
    "adversarial": ("adversarial", _adapter("adversarial", "attack_type", lambda r: _wrong(r) and r["attack_type"] != "none", lambda r: (["adversarial_manipulation"], "adversarial_manipulation"), "clean_input", ["attack_type", "target_label", "intent"], base_mask=lambda d: d["attack_type"] == "none")),
    "choice_overlap": ("choice_overlap", _adapter("choice_overlap", "choice_set_type", _wrong, lambda r: (["semantic_overlap"], "semantic_overlap") if r["choice_set_type"] == "overlapping" else ([], None), "text", ["choice_set_type", "number_of_choices", "choice_set_id", "family"])),
}


def stress_failures(name: str, results: pd.DataFrame, source_file: str, high: float = HIGH_CONFIDENCE_THRESHOLD, low: float = LOW_CONFIDENCE_THRESHOLD) -> list[dict[str, Any]]:
    return ADAPTERS[name][1](results, source_file, high, low)


# --- assembling ----------------------------------------------------------------------------------


def validate_failures(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    for c in REQUIRED_FIELDS:
        if df[c].isna().any():
            raise ValueError(f"required field {c!r} is null in {int(df[c].isna().sum())} row(s)")
    unknown = {t for tags in df["failure_tags"] for t in tags} - set(FAILURE_TAGS)
    if unknown:
        raise ValueError(f"unknown failure tags: {sorted(unknown)}")
    if df["failure_id"].duplicated().any():
        raise ValueError("duplicate failure_id")


def assemble(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per failure_id (a repeat of the same failure keeps its latest record), in a stable order."""
    df = pd.DataFrame(records, columns=COLUMNS)
    df = df.drop_duplicates("failure_id", keep="last").sort_values("failure_id").reset_index(drop=True)
    validate_failures(df)
    return df


def summarize(df: pd.DataFrame, missing_experiments: list[str], high: float, low: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    tag_counts = {t: int(sum(t in tags for tags in df["failure_tags"])) for t in FAILURE_TAGS}
    unclassified = df[df["failure_tags"].map(len) == 0]
    top = df[df["failure_tags"].map(lambda t: "high_confidence_wrong" in t)].sort_values(["confidence", "failure_id"], ascending=[False, True]).head(10)
    return {
        "total_failures": len(df),
        "by_experiment": {k: int(v) for k, v in df["experiment"].value_counts().sort_index().items()},
        "by_provider": {k: int(v) for k, v in df["provider"].value_counts().sort_index().items()},
        "by_failure_tag": tag_counts,
        "n_with_multiple_tags": int((df["failure_tags"].map(len) > 1).sum()),
        "n_with_primary_failure_type": int(df["primary_failure_type"].notna().sum()),
        "n_unclassified_no_tag": len(unclassified),
        "unclassified_by_experiment": {k: int(v) for k, v in unclassified["experiment"].value_counts().sort_index().items()},
        "unclassified_note": "no confidence tag and no experiment-established cause: the evidence does not classify these",
        "high_confidence_wrong_examples": [{"failure_id": r["failure_id"], "experiment": r["experiment"], "provider": r["provider"], "confidence": r["confidence"], "ground_truth": r["ground_truth"], "prediction": r["prediction"], "input": r["input"]} for _, r in top.iterrows()],
        "experiments_with_no_recorded_results": missing_experiments,
        "thresholds": {"high_confidence": high, "low_confidence": low, "note": "configurable slicing thresholds, not scientifically optimal"},
        **(extra or {}),
    }


def run_failures(baseline_dir: Path = BASELINE_DIR, phase2_dir: Path = PHASE2_RESULTS_DIR, out_dir: Path = FAILURES_DIR, verify: bool = True, high: float = HIGH_CONFIDENCE_THRESHOLD, low: float = LOW_CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    if verify:
        errors, _ = verify_baseline(baseline_dir / MANIFEST_NAME)
        if errors:
            raise BaselineError(f"Baseline failed verification: {errors}")
    existing = [str(out_dir / n) for n in OUTPUT_FILES if (out_dir / n).exists()]
    if existing:
        raise BaselineError(f"Refusing to overwrite existing output(s): {existing}")
    source = baseline_dir / CANONICAL_NAME
    records = baseline_failures(pd.read_parquet(source), str(source), high, low)
    missing, used_files = [], {str(source): sha256_file(source)}
    for name, (results_dir, _) in ADAPTERS.items():
        files = sorted((phase2_dir / results_dir).glob("raw_results_*.parquet"))
        if not files:
            missing.append(name)
            continue
        for f in files:
            records += stress_failures(name, pd.read_parquet(f), str(f), high, low)
            used_files[str(f)] = sha256_file(f)
    df = assemble(records)
    summary = summarize(df, missing, high, low, {"source_files": used_files})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as fh:
        for rec in df.to_dict("records"):
            fh.write(json.dumps({k: (v if not hasattr(v, "item") else v.item()) for k, v in rec.items()}, ensure_ascii=False, default=str, sort_keys=True) + "\n")
    flat = df.assign(failure_tags=df["failure_tags"].map("|".join), metadata=df["metadata"].map(lambda m: json.dumps(m, default=str, sort_keys=True)))
    flat.to_csv(out_dir / "failures.csv", index=False)
    (out_dir / "failures_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
