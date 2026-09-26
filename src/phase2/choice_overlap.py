"""Phase II Step 11: choice-overlap stress.

Phase I varied how MANY choices there are. This asks what happens when the choices themselves become
MORE SIMILAR at the same K.

For each source example with true intent `a` and each K in {3, 4}, the SAME source is evaluated against
two frozen choice sets that both contain `a`:

    overlapping   a + (K-1) other real Bitext labels from a's own intent family (e.g. refund vs
                  track_refund vs check_refund_policy)
    distinct      a + (K-1) real labels from K-1 DIFFERENT other families, none in a's family

All labels are real taxonomy labels; none are invented to inflate K. Families are a hand-defined
coarse grouping of the 27 Bitext intents (below). Choice order within a set is a seeded shuffle.
Phase I's K-scaling results remain the existing baseline for "more choices"; nothing here repeats it.

Top-1/top-2 margin is computed only from a recorded full probability distribution, never invented.
"""

import json
from typing import Any

import pandas as pd

from phase2.stress import (
    PHASE2_RESULTS_DIR,
    STRESS_DIR,
    ExperimentSpec,
    StressError,
    pair_results,
    paired_metrics,
    seeded_rng,
    select_bitext_sources,
    source_selection_seed_fields,
    stable_int,
    top_margin,
)

NAME = "choice_overlap"
DATASET_DIR = STRESS_DIR / "choice_overlap"
RESULTS_DIR = PHASE2_RESULTS_DIR / "choice_overlap"
N_SOURCES = 100
K_VALUES = (3, 4)
SET_TYPES = ("distinct", "overlapping")
FAMILIES: dict[str, list[str]] = {
    "refund": ["get_refund", "track_refund", "check_refund_policy"],
    "order": ["cancel_order", "change_order", "place_order", "track_order", "check_cancellation_fee"],
    "account": ["create_account", "delete_account", "edit_account", "switch_account", "registration_problems", "recover_password"],
    "invoice": ["check_invoice", "get_invoice"],
    "payment": ["check_payment_methods", "payment_issue"],
    "shipping": ["change_shipping_address", "set_up_shipping_address", "delivery_options", "delivery_period"],
    "contact": ["contact_customer_service", "contact_human_agent", "complaint"],
    "newsletter": ["newsletter_subscription"],
    "review": ["review"],
}
FAMILY_OF = {intent: fam for fam, members in FAMILIES.items() for intent in members}
MIN_FAMILY_SIZE = max(K_VALUES)  # sources are drawn from families large enough for the biggest K


def family_of(intent: str) -> str:
    if intent not in FAMILY_OF:
        raise StressError(f"intent {intent!r} has no family")
    return FAMILY_OF[intent]


def overlapping_set(source_id: str, truth: str, k: int) -> list[str]:
    fam = family_of(truth)
    others = sorted(i for i in FAMILIES[fam] if i != truth)
    if len(others) < k - 1:
        raise StressError(f"family {fam!r} too small for K={k}")
    picked = seeded_rng("overlap-pick", source_id, k).sample(others, k - 1)
    return sorted([truth, *picked])


def distinct_set(source_id: str, truth: str, k: int) -> list[str]:
    """K-1 labels from K-1 different families, all outside the truth's family."""
    fam = family_of(truth)
    families = sorted(f for f in FAMILIES if f != fam)
    chosen_families = seeded_rng("distinct-families", source_id, k).sample(families, k - 1)
    labels = [seeded_rng("distinct-label", source_id, k, f).choice(sorted(FAMILIES[f])) for f in chosen_families]
    return sorted([truth, *labels])


def build_set(source_id: str, truth: str, k: int, set_type: str) -> list[str]:
    labels = {"overlapping": overlapping_set, "distinct": distinct_set}[set_type](source_id, truth, k)
    seeded_rng("order", source_id, k, set_type).shuffle(labels)  # order is seeded, not alphabetical
    return labels


def validate_choice_set(truth: str, labels: list[str], k: int, set_type: str) -> None:
    fam = family_of(truth)
    if truth not in labels or len(labels) != k or len(set(labels)) != k:
        raise StressError(f"invalid choice set for truth {truth}: {labels}")
    same_family = [x for x in labels if x != truth and family_of(x) == fam]
    if set_type == "overlapping" and len(same_family) != k - 1:
        raise StressError(f"overlapping set must be all in family {fam}: {labels}")
    if set_type == "distinct" and (same_family or len({family_of(x) for x in labels}) != k):
        raise StressError(f"distinct set must span {k} different families and none may be in {fam}: {labels}")


def rows_for_source(source_id: str, text: str, truth: str) -> list[dict[str, Any]]:
    rows = []
    for k in K_VALUES:
        for set_type in SET_TYPES:
            labels = build_set(source_id, truth, k, set_type)
            validate_choice_set(truth, labels, k, set_type)
            csid = f"{source_id}:k{k}:{set_type}"
            rows.append({"variant_id": csid, "source_example_id": source_id, "choice_set_id": csid, "choice_set_type": set_type, "number_of_choices": k, "text": text, "candidates": labels, "ground_truth": truth, "expected_label": truth, "intent": truth, "family": family_of(truth)})
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [r for r in select_bitext_sources(1000) if len(FAMILIES[family_of(r.ground_truth)]) >= MIN_FAMILY_SIZE]
    chosen = sorted(eligible, key=lambda r: stable_int("overlap-source", r.id))[:N_SOURCES]
    rows: list[dict[str, Any]] = []
    for r in sorted(chosen, key=lambda r: r.id):
        rows.extend(rows_for_source(r.id, r.text, r.ground_truth))
    meta = {
        "experiment": NAME,
        "source": "data/samples/bitext_main.jsonl via select_bitext_sources",
        **source_selection_seed_fields("The 1000-example pool is drawn with it. Choice-set construction and ordering use seeded_rng keyed by constant labels (overlap-pick, distinct-families, distinct-label, order) + source_id + K; the source ranking is a stable_int hash. There is no separate numeric seed."),
        "n_sources_requested": N_SOURCES,
        "k_values": list(K_VALUES),
        "set_types": list(SET_TYPES),
        "families": FAMILIES,
        "construction": "overlapping = truth + K-1 labels from the truth's family; distinct = truth + K-1 labels from K-1 different other families; seeded; all real Bitext labels",
        "phase1_baseline": "Phase I choice-scaling / Bitext K=5 results are the existing baseline; not repeated here",
        "generation_method": "deterministic seeded construction; no model",
        "review_status": "pending_human_review",
    }
    return rows, meta


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    ex = [{"source_example_id": r["source_example_id"], "k": r["number_of_choices"], "type": r["choice_set_type"], "truth": r["ground_truth"], "candidates": r["candidates"]} for _, r in df.head(8).iterrows()]
    return {
        "n_source_examples": df["source_example_id"].nunique(),
        "n_rows": len(df),
        "n_choice_sets": df["choice_set_id"].nunique(),
        "counts_by_k_and_type": {f"k{k}/{t}": int(n) for (k, t), n in df.groupby(["number_of_choices", "choice_set_type"]).size().items()},
        "sources_by_family": df[df["choice_set_type"] == "overlapping"].groupby("family").size().div(len(K_VALUES)).astype(int).to_dict(),
        "example_candidate_sets": ex,
    }


def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    probs = out["probabilities_json"] if "probabilities_json" in out else [None] * len(out)
    out["margin"] = [top_margin(json.loads(s)) if isinstance(s, str) else None for s in probs]
    return out


def analyze(results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    df = add_margin(results)
    rows = []
    for (provider, k, stype), grp in df.groupby(["provider", "number_of_choices", "choice_set_type"], sort=True):
        ok = grp[grp["prediction"].notna()]
        margins = ok["margin"].dropna()
        rows.append(
            {
                "provider": provider,
                "number_of_choices": k,
                "choice_set_type": stype,
                "n": len(grp),
                "n_predicted": len(ok),
                "accuracy": float(ok["correct"].dropna().astype(bool).mean()) if ok["correct"].notna().any() else None,
                "mean_confidence": float(ok["confidence"].mean()) if len(ok) else None,
                "mean_top1_top2_margin": float(margins.mean()) if len(margins) else None,
                "n_with_margin": len(margins),
                "margin_note": "available" if len(margins) else "unavailable: full probabilities not recorded",
            }
        )
    by_set = pd.DataFrame(rows)
    on = ("provider", "source_example_id", "number_of_choices")
    distinct, overlapping = df[df["choice_set_type"] == "distinct"], df[df["choice_set_type"] == "overlapping"]
    paired = pair_results(distinct, overlapping, on=on)
    diffs = []
    for (provider, k), grp in paired.groupby(["provider", "number_of_choices"], sort=True):
        m = paired_metrics(grp)
        diffs.append({"provider": provider, "number_of_choices": k, "n_pairs": m["n_pairs"], "accuracy_distinct": m.get("accuracy_base"), "accuracy_overlapping": m.get("accuracy_variant"), "accuracy_difference_overlapping_minus_distinct": m.get("accuracy_delta"), "decision_flip_rate": m.get("decision_flip_rate"), "mean_confidence_delta_overlapping_minus_distinct": m.get("mean_confidence_delta")})
    summary = {
        "experiment": NAME,
        "n_results": len(df),
        "pairing": "the same source example under its distinct set vs its overlapping set (same K, same provider); the true label is in both",
        "interpretation": "not concluded here; read differences with n_pairs",
    }
    pairs = paired[["provider", "source_example_id", "number_of_choices", "expected_label", "prediction_base", "prediction_variant", "confidence_base", "confidence_variant", "correct_base", "correct_variant"]].rename(columns={"prediction_base": "distinct_prediction", "prediction_variant": "overlapping_prediction", "confidence_base": "distinct_confidence", "confidence_variant": "overlapping_confidence", "correct_base": "distinct_correct", "correct_variant": "overlapping_correct"})
    return {"choice_overlap_by_set": by_set, "choice_overlap_paired": pd.DataFrame(diffs), "choice_overlap_pairs": pairs}, summary


SPEC = ExperimentSpec(name=NAME, dataset_dir=DATASET_DIR, results_dir=RESULTS_DIR, build=build, describe=describe, analyze=analyze)
