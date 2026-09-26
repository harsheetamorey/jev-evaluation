"""Phase II Step 9: out-of-distribution evaluation ("none of the choices is correct").

Question: what happens when none of the allowed choices is actually correct, and does an explicit
fallback choice change the behaviour?

Populations (kept separate, never mixed into one accuracy figure):
  in_domain   Bitext support messages (the matched CONTROL population)
  ood         unrelated text with NO correct label in the support taxonomy

OOD source: the en-US voice-assistant utterances of the frozen MASSIVE sample already in the repo
(music, weather, alarms, ...), filtered so none matches a support-intent keyword. This is the
"another suitable source already available" option: nothing is downloaded. AG News (news headlines)
would be a stronger, more clearly public OOD source and can be swapped in after approval to download it.

Matched design: OOD example i is given the SAME five candidate intents as in-domain control example i.

Two contracts, identical in everything except the candidate list:
  forced_choice      the five support intents only
  explicit_fallback  the same five plus "other"

The experiment does NOT test whether Jev "detects OOD". It tests whether offering an explicit fallback
choice changes the outcome when the taxonomy lacks the answer.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_loaders.massive import load_sample
from phase2.context_relevance import KEYWORDS
from phase2.stress import (
    PHASE2_RESULTS_DIR,
    REVIEW_FIELDS,
    STRESS_DIR,
    ExperimentSpec,
    StressError,
    seeded_rng,
    select_bitext_sources,
)

NAME = "ood"
DATASET_DIR = STRESS_DIR / "ood"
RESULTS_DIR = PHASE2_RESULTS_DIR / "ood"
N_PER_POPULATION = 100
OOD_SEED = 9
OTHER_LABEL = "other"
CONTRACTS = ("forced_choice", "explicit_fallback")
POPULATIONS = ("in_domain", "ood")
FALSE_CONFIDENCE_THRESHOLDS = (0.80, 0.90, 0.95)
MASSIVE_SAMPLE = "aligned_250"
MASSIVE_SAMPLE_FILE = Path("data/samples/massive_aligned_250.jsonl")
_SUPPORT_TERMS = sorted({k.strip() for kws in KEYWORDS.values() for k in kws} | {"order", "invoice", "refund", "account", "payment", "delivery", "shipping", "complain"})


def is_support_related(text: str) -> bool:
    t = f" {text.lower()} "
    return any(term in t for term in _SUPPORT_TERMS)


def ood_pool() -> list[dict[str, str]]:
    """Frozen MASSIVE en-US utterances with >= 3 words and no support-intent keyword."""
    rows = [r for r in load_sample(MASSIVE_SAMPLE) if r.locale == "en-US" and len(r.text.split()) >= 3 and not is_support_related(r.text)]
    return sorted(({"source_id": f"massive:{r.locale}:{r.id}", "text": r.text, "massive_intent": r.ground_truth} for r in rows), key=lambda x: x["source_id"])


def contract_candidates(base: list[str], contract: str) -> list[str]:
    if contract == "forced_choice":
        return list(base)
    if contract == "explicit_fallback":
        return [*base, OTHER_LABEL]
    raise StressError(f"unknown contract {contract!r}")


def rows_for_pair(control_id: str, control_text: str, control_truth: str, choices: list[str], ood: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for contract in CONTRACTS:
        cands = contract_candidates(choices, contract)
        common = {"contract_type": contract, "candidates": cands, "matched_control_id": control_id, "n_base_choices": len(choices)}
        rows.append(
            {
                **common,
                "variant_id": f"in_domain:{control_id}:{contract}",
                "source_example_id": control_id,
                "population": "in_domain",
                "text": control_text,
                "ground_truth": control_truth,  # the real intent, under both contracts
                "expected_label": control_truth,
            }
        )
        rows.append(
            {
                **common,
                "variant_id": f"ood:{ood['source_id']}:{contract}",
                "source_example_id": ood["source_id"],
                "population": "ood",
                "text": ood["text"],
                # No support intent is correct. Forced choice has no correct answer at all; the fallback
                # contract's "other" is the only defensible label, recorded but NOT read as OOD detection.
                "ground_truth": OTHER_LABEL if contract == "explicit_fallback" else None,
                "expected_label": OTHER_LABEL if contract == "explicit_fallback" else None,
                "massive_intent": ood["massive_intent"],
            }
        )
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls = sorted(select_bitext_sources(N_PER_POPULATION), key=lambda r: r.id)
    pool = ood_pool()
    if len(pool) < N_PER_POPULATION:
        raise StressError(f"OOD pool has only {len(pool)} usable utterances, need {N_PER_POPULATION}")
    oods = sorted(seeded_rng("ood-pick", OOD_SEED).sample(pool, N_PER_POPULATION), key=lambda x: x["source_id"])
    rows: list[dict[str, Any]] = []
    for c, o in zip(controls, oods, strict=True):
        rows.extend(rows_for_pair(c.id, c.text, c.ground_truth, c.choices, o))
    meta = {
        "experiment": NAME,
        "ood_source": "MASSIVE en-US utterances from the frozen sample data/samples/massive_aligned_250.jsonl",
        "ood_source_file_sha256": hashlib.sha256(MASSIVE_SAMPLE_FILE.read_bytes()).hexdigest(),
        "ood_dataset_revision": "not_recorded (content pinned by ood_source_file_sha256)",
        "ood_pool_size_after_filter": len(pool),
        "ood_filter": "en-US, >=3 words, no support-intent keyword",
        "control_source": "data/samples/bitext_main.jsonl via select_bitext_sources",
        "n_per_population": N_PER_POPULATION,
        "ood_source_ids": [o["source_id"] for o in oods],
        "control_source_ids": [c.id for c in controls],
        "matching": "OOD example i uses the same candidate set as control example i",
        "contracts": {"forced_choice": "5 support intents", "explicit_fallback": f"same 5 + '{OTHER_LABEL}' appended; instructions identical"},
        "false_confidence_thresholds": list(FALSE_CONFIDENCE_THRESHOLDS),
        "seed": OOD_SEED,
        "alternative_source_pending_approval": "AG News (requires download; not fetched)",
        **REVIEW_FIELDS,
    }
    return rows, meta


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    ex = df[(df["population"] == "ood") & (df["contract_type"] == "forced_choice")].head(10)
    return {
        "n_rows": len(df),
        "counts_by_population_and_contract": {f"{p}/{c}": int(n) for (p, c), n in df.groupby(["population", "contract_type"]).size().items()},
        "n_ood_examples": df[df["population"] == "ood"]["source_example_id"].nunique(),
        "n_control_examples": df[df["population"] == "in_domain"]["source_example_id"].nunique(),
        "ood_examples": [{"id": r["source_example_id"], "text": r["text"], "massive_intent": r["massive_intent"]} for _, r in ex.iterrows()],
        "fallback_label": OTHER_LABEL,
    }


def confidence_summary(conf: pd.Series) -> dict[str, Any]:
    c = conf.dropna()
    if c.empty:
        return {"n_with_confidence": 0}
    return {"n_with_confidence": len(c), "mean_confidence": float(c.mean()), "median_confidence": float(c.median()), "p10_confidence": float(c.quantile(0.1)), "p90_confidence": float(c.quantile(0.9))}


def false_confidence_rates(conf: pd.Series, thresholds: tuple[float, ...] = FALSE_CONFIDENCE_THRESHOLDS) -> dict[str, float | None]:
    """Share of predictions with confidence >= t. For OOD forced-choice every prediction is wrong by construction,
    so this IS the false-confidence rate. Reported at several thresholds; none is privileged."""
    c = conf.dropna().round(6)
    return {f"share_conf_ge_{t:.2f}": (float((c >= t).mean()) if len(c) else None) for t in thresholds}


def ks_statistic(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def analyze(results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    need = {"provider", "population", "contract_type", "prediction", "confidence"}
    if missing := need - set(results.columns):
        raise StressError(f"results are missing columns: {sorted(missing)}")
    forced_rows, fallback_rows, label_rows, bin_rows, compare_rows = [], [], [], [], []
    for provider, prov in results.groupby("provider"):
        for pop in POPULATIONS:
            forced = prov[(prov["population"] == pop) & (prov["contract_type"] == "forced_choice")]
            ok = forced[forced["prediction"].notna()]
            row = {"provider": provider, "population": pop, "n": len(forced), "n_predicted": len(ok), **confidence_summary(ok["confidence"]), **false_confidence_rates(ok["confidence"])}
            if pop == "in_domain" and ok["correct"].notna().any():
                row["accuracy"] = float(ok["correct"].dropna().astype(bool).mean())
            forced_rows.append(row)
            for label, n in ok["prediction"].value_counts().items():
                label_rows.append({"provider": provider, "population": pop, "contract_type": "forced_choice", "predicted_label": label, "count": int(n), "share": float(n / len(ok))})
            fb = prov[(prov["population"] == pop) & (prov["contract_type"] == "explicit_fallback")]
            fok = fb[fb["prediction"].notna()]
            is_other = fok["prediction"] == OTHER_LABEL
            fallback_rows.append(
                {
                    "provider": provider,
                    "population": pop,
                    "n": len(fb),
                    "n_predicted": len(fok),
                    "other_selection_rate": float(is_other.mean()) if len(fok) else None,
                    "non_other_rate": float((~is_other).mean()) if len(fok) else None,
                    "n_other": int(is_other.sum()),
                    "n_non_other": int((~is_other).sum()),
                    "mean_confidence_when_other": float(fok.loc[is_other, "confidence"].mean()) if is_other.any() else None,
                    "mean_confidence_when_forced_into_in_domain_intent": float(fok.loc[~is_other, "confidence"].mean()) if (~is_other).any() else None,
                    **{f"non_other_share_conf_ge_{t:.2f}": (float(((~is_other) & (fok["confidence"].round(6) >= t)).sum() / len(fok)) if len(fok) else None) for t in FALSE_CONFIDENCE_THRESHOLDS},
                }
            )
            for contract, grp in (("forced_choice", ok), ("explicit_fallback", fok)):
                for b in range(10):
                    sel = grp[(grp["confidence"].round(6) * 10).clip(upper=9.999999).astype(float).map(np.floor) == b]
                    bin_rows.append({"provider": provider, "population": pop, "contract_type": contract, "bin_label": f"[{b/10:.1f}, {(b+1)/10:.1f}{']' if b == 9 else ')'}", "sample_count": len(sel)})
        for contract in CONTRACTS:
            a = prov[(prov["population"] == "in_domain") & (prov["contract_type"] == contract)]["confidence"]
            o = prov[(prov["population"] == "ood") & (prov["contract_type"] == contract)]["confidence"]
            compare_rows.append({"provider": provider, "contract_type": contract, "n_in_domain": int(a.notna().sum()), "n_ood": int(o.notna().sum()), "mean_conf_in_domain": float(a.mean()) if a.notna().any() else None, "mean_conf_ood": float(o.mean()) if o.notna().any() else None, "ks_statistic_conf_in_domain_vs_ood": ks_statistic(a, o)})
    tables = {
        "ood_forced_choice": pd.DataFrame(forced_rows),
        "ood_fallback": pd.DataFrame(fallback_rows),
        "ood_label_distribution": pd.DataFrame(label_rows),
        "ood_confidence_bins": pd.DataFrame(bin_rows),
        "ood_confidence_in_domain_vs_ood": pd.DataFrame(compare_rows),
    }
    summary = {
        "experiment": NAME,
        "n_results": len(results),
        "false_confidence_definition": "OOD forced-choice: share of predictions with confidence >= threshold (every forced choice is wrong by construction); thresholds 0.80/0.90/0.95, none privileged",
        "caveat": "selecting 'other' is not evidence that Jev detects OOD; this tests whether an explicit fallback choice helps when the taxonomy lacks the answer",
        "interpretation": "not concluded here; read rates together with n",
    }
    return tables, summary


SPEC = ExperimentSpec(name=NAME, dataset_dir=DATASET_DIR, results_dir=RESULTS_DIR, build=build, describe=describe, analyze=analyze)
