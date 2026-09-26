"""Phase II Step 7: context pollution.

Question: does adding irrelevant state leave Jev's decision unchanged, or can irrelevant context
actively change it?

Every target message is sent in the same structured shape at every level, so the ONLY thing that
varies is the amount of irrelevant history:

    none      {"message": target}
    small ... {"message": target, "history": [irrelevant messages ...]}

The irrelevant history is real, natural text from a different domain: the en-US utterances of the
MASSIVE voice-assistant sample (music, weather, calendar, news ...), which have nothing to do with
customer-support intents. Each source example gets ONE seeded shuffle of that pool and every level
takes a prefix of it, so levels are nested (medium contains small) and size strictly increases.
Every fragment's source id is stored. No label string is inserted.

Context size is a configurable CHARACTER budget. The TypeSafe SDK exposes no context-window metadata,
so no limit is assumed: `very_large` (8000 chars) is only the largest configured budget (bounded by the
size of the irrelevant pool). It is NOT described as, or verified to be, near the supported context window. Actual input size is measured
from the token usage recorded at evaluation time.
"""

import json
from typing import Any

import pandas as pd

from dataset_loaders.massive import load_sample
from phase2.stress import (
    PHASE2_RESULTS_DIR,
    REVIEW_FIELDS,
    STRESS_DIR,
    ExperimentSpec,
    StressError,
    grouped_paired_metrics,
    pair_results,
    paired_metrics,
    seeded_rng,
    select_bitext_sources,
)

NAME = "context_pollution"
DATASET_DIR = STRESS_DIR / "context_pollution"
RESULTS_DIR = PHASE2_RESULTS_DIR / "context_pollution"
N_SOURCES = 100
CONTEXT_SEED = 7
POOL_SAMPLE = "aligned_250"
POOL_LOCALE = "en-US"
# Character budgets of the irrelevant history per level (configurable; no SDK limit is assumed).
CONTEXT_BUDGET_CHARS = {"none": 0, "small": 300, "medium": 1500, "large": 4000, "very_large": 8000}  # very_large = 8000 chars; NOT a claim about Jev's real limit
LEVELS = tuple(CONTEXT_BUDGET_CHARS)


def irrelevant_pool() -> list[dict[str, str]]:
    """The unrelated-domain fragments: MASSIVE en-US utterances, with stable source ids."""
    rows = [r for r in load_sample(POOL_SAMPLE) if r.locale == POOL_LOCALE]
    return sorted(({"source_id": f"massive:{r.locale}:{r.id}", "text": r.text} for r in rows), key=lambda x: x["source_id"])


def leaks_label(history_texts: list[str], candidates: list[str]) -> bool:
    """True if any history text contains a candidate label (raw or with spaces)."""
    blob = " ".join(history_texts).lower()
    return any(c.lower() in blob or c.lower().replace("_", " ") in blob for c in candidates)


def build_history(pool: list[dict[str, str]], source_id: str, budget_chars: int, seed: int = CONTEXT_SEED, avoid_labels: list[str] | None = None) -> list[dict[str, str]]:
    """Prefix of this source's seeded shuffle whose total text length stays within the budget.

    Fragments that mention one of `avoid_labels` are skipped (deterministically), so the history
    never names a candidate intent, even by natural coincidence.
    """
    order = list(pool)
    seeded_rng("context-order", seed, source_id).shuffle(order)
    if avoid_labels:
        order = [f for f in order if not leaks_label([f["text"]], avoid_labels)]
    out, used = [], 0
    for frag in order:
        if used + len(frag["text"]) > budget_chars:
            break
        out.append(frag)
        used += len(frag["text"])
    return out


def rows_for_source(source_id: str, text: str, choices: list[str], truth: str, pool: list[dict[str, str]], budgets: dict[str, int] = CONTEXT_BUDGET_CHARS, seed: int = CONTEXT_SEED) -> list[dict[str, Any]]:
    rows = []
    for level, budget in budgets.items():
        frags = build_history(pool, source_id, budget, seed, avoid_labels=choices) if budget else []
        history = [f["text"] for f in frags]
        payload: dict[str, Any] = {"message": text} if not history else {"message": text, "history": history}
        if leaks_label(history, choices):
            raise StressError(f"context for {source_id}/{level} contains a candidate label")
        rows.append(
            {
                "variant_id": f"{source_id}:{level}",
                "source_example_id": source_id,
                "context_level": level,
                "context_budget_chars": budget,
                "base_message": text,
                "text": text,
                "state_payload": payload,
                "polluted_state": payload,
                "context_source_ids": [f["source_id"] for f in frags],
                "n_history_items": len(history),
                "history_chars": sum(len(h) for h in history),
                "input_chars": len(json.dumps(payload, ensure_ascii=False)),
                "expected_label": truth,
                "ground_truth": truth,
                "candidates": choices,
                "intent": truth,
            }
        )
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = irrelevant_pool()
    rows: list[dict[str, Any]] = []
    for r in sorted(select_bitext_sources(N_SOURCES), key=lambda r: r.id):
        rows.extend(rows_for_source(r.id, r.text, r.choices, r.ground_truth, pool))
    meta = {
        "experiment": NAME,
        "source": "targets: data/samples/bitext_main.jsonl via select_bitext_sources; irrelevant context: MASSIVE en-US aligned_250 utterances",
        "n_sources_requested": N_SOURCES,
        "context_seed": CONTEXT_SEED,
        "context_budget_chars": CONTEXT_BUDGET_CHARS,
        "irrelevant_pool_size": len(pool),
        "irrelevant_pool_total_chars": sum(len(p["text"]) for p in pool),
        "state_shape": {"none": {"message": "<target>"}, "others": {"message": "<target>", "history": ["<irrelevant text>", "..."]}},
        "context_limit_note": "TypeSafe SDK exposes no context-window metadata; budgets are configurable; very_large (8000 chars) is only the largest configured budget and is NOT claimed to approach Jev's real limit",
        **REVIEW_FIELDS,
    }
    return rows, meta


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    dist = df.groupby("context_level")["input_chars"].describe()[["min", "50%", "max"]].round(0)
    ex = []
    for level in ("none", "small", "medium"):
        r = df[df["context_level"] == level].iloc[0]
        ex.append({"variant_id": r["variant_id"], "level": level, "message": r["base_message"], "n_history_items": int(r["n_history_items"]), "history_preview": list(r["state_payload"].get("history", []))[:2], "expected_label": r["expected_label"]})
    return {
        "n_source_examples": df["source_example_id"].nunique(),
        "n_rows": len(df),
        "counts_by_context_level": df["context_level"].value_counts().to_dict(),
        "input_chars_distribution_by_level": {lvl: {k: int(v) for k, v in row.items()} for lvl, row in dist.iterrows()},
        "history_items_by_level_mean": df.groupby("context_level")["n_history_items"].mean().round(1).to_dict(),
        "representative_examples": ex,
    }


def analyze(results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not (results["context_level"] == "none").any():
        raise StressError("No 'none' rows; paired comparison needs each source's no-context row.")
    base = results[results["context_level"] == "none"]
    polluted = results[results["context_level"] != "none"]
    paired = pair_results(base, polluted)
    rows = []
    for (provider, level), grp in paired.groupby(["provider", "context_level"], sort=False):
        m = paired_metrics(grp)
        ok = grp[grp["latency_ms_base"].notna() & grp["latency_ms_variant"].notna()]
        rows.append(
            {
                "provider": provider,
                "context_level": level,
                **m,
                "mean_latency_ms_base": float(ok["latency_ms_base"].mean()) if len(ok) else None,
                "mean_latency_ms_variant": float(ok["latency_ms_variant"].mean()) if len(ok) else None,
                "mean_latency_delta_ms": float((ok["latency_ms_variant"] - ok["latency_ms_base"]).mean()) if len(ok) else None,
                "mean_input_tokens_variant": float(grp["input_tokens_variant"].mean()) if "input_tokens_variant" in grp and grp["input_tokens_variant"].notna().any() else None,
            }
        )
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    by_level = pd.DataFrame(rows)
    by_level = by_level.sort_values(["provider", "context_level"], key=lambda c: c.map(order) if c.name == "context_level" else c).reset_index(drop=True)
    sizes = results.groupby(["provider", "context_level"], sort=False).agg(n=("input_chars", "size"), mean_input_chars=("input_chars", "mean"), mean_history_items=("n_history_items", "mean"), mean_input_tokens=("input_tokens", "mean")).reset_index()
    pairs = paired[["provider", "source_example_id", "context_level", "expected_label", "prediction_base", "prediction_variant", "confidence_base", "confidence_variant", "correct_base", "correct_variant"]].copy()
    pairs["confidence_delta"] = pairs["confidence_variant"] - pairs["confidence_base"]
    summary = {
        "experiment": NAME,
        "n_results": len(results),
        "pairing": "each polluted row vs its own source's no-context row (same provider)",
        "context_budget_chars": CONTEXT_BUDGET_CHARS,
        "interpretation": "not concluded here; whether longer context matters is read from accuracy_delta / flip rate with n_pairs",
    }
    return {"context_by_level": by_level, "context_input_sizes": sizes, "context_pairs": pairs}, summary


SPEC = ExperimentSpec(name=NAME, dataset_dir=DATASET_DIR, results_dir=RESULTS_DIR, build=build, describe=describe, analyze=analyze)
