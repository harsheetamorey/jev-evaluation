"""Phase II Step 8: relevant vs irrelevant context, on paired triplets.

Question: for the same underlying decision, how do NO context, IRRELEVANT context and RELEVANT context
compare, and which individual decisions move between right and wrong?

Each triplet is hand-authored (once, frozen, pending human review) around a deliberately vague target
message such as "It happened again.":

    no_context          {"message": target}
    irrelevant_context  {"message": target, "history": [plausible history unrelated to the decision]}
    relevant_context    {"message": target, "history": [history that genuinely resolves the intent]}

Candidates are the real Bitext choice set for the expected intent. How relevance is established:
  1. the target is written to be vague between the candidates on its own;
  2. the relevant history names the specific detail that selects the expected intent (`relevance_cues`);
  3. an automated validator enforces: the cues appear in the relevant history and NOT in the target or the
     irrelevant history; no OTHER candidate's keywords appear in the relevant history; no candidate's
     keywords appear in the target or the irrelevant history; histories are length-comparable;
  4. every triplet is left for human review (`review_status`).
This is a check on construction, not proof of relevance; it is why the set is small and inspectable.

Reuses the state shape of Step 7 (`message` / `history`) but is kept separate from the generic
context-pollution results.
"""

from typing import Any

import pandas as pd

from dataset_loaders.bitext import load_sample
from phase2.stress import (
    PHASE2_RESULTS_DIR,
    REVIEW_FIELDS,
    STRESS_DIR,
    ExperimentSpec,
    StressError,
    pair_results,
    paired_metrics,
    seed_fields,
)

NAME = "context_relevance"
DATASET_DIR = STRESS_DIR / "context_relevance"
RESULTS_DIR = PHASE2_RESULTS_DIR / "context"
CONDITIONS = ("no_context", "irrelevant_context", "relevant_context")
LENGTH_RATIO_RANGE = (0.6, 1.7)  # irrelevant/relevant history length must fall in this range

KEYWORDS: dict[str, list[str]] = {
    "cancel_order": ["cancel"], "change_order": ["change", "modify", "amend"], "change_shipping_address": ["address"],
    "check_cancellation_fee": [" fee"], "check_invoice": ["invoice"], "check_payment_methods": ["payment method", "ways to pay"],
    "complaint": ["complain"], "contact_customer_service": ["customer service", "customer support"], "contact_human_agent": ["human", "agent"],
    "create_account": ["create an account", "sign up", "sign-up"], "delete_account": ["delete", "close my account"],
    "delivery_options": ["delivery option", "shipping option"], "delivery_period": ["arrive", "how long"], "edit_account": ["edit"],
    "get_invoice": ["invoice"], "get_refund": ["refund", "money back"], "newsletter_subscription": ["newsletter"],
    "payment_issue": ["payment", "charged"], "place_order": ["place an order", "buy"], "recover_password": ["password"],
    "registration_problems": ["registration", "register"], "review": ["review", "rate "], "set_up_shipping_address": ["address"],
    "switch_account": ["switch"], "track_order": ["track", "where is my order"], "track_refund": ["refund status"],
}

# (expected intent, vague target, relevant history, irrelevant history, cues that must appear only in the relevant history)
TRIPLETS: list[tuple[str, str, str, str, list[str]]] = [
    ("payment_issue", "It happened again.", "Earlier the customer said their card payment was declined at checkout.", "Earlier the customer said they prefer to be contacted in the afternoon.", ["declined"]),
    ("track_order", "Any news yet?", "Earlier the customer said the blue jacket they ordered last Monday has not shown up.", "Earlier the customer said they recently upgraded to a newer phone model.", ["has not shown up"]),
    ("track_refund", "Any update on this?", "Earlier the customer said they sent the shoes back two weeks ago and are waiting for the money.", "Earlier the customer said they are based in the Pacific time zone.", ["waiting for the money"]),
    ("get_refund", "I want that sorted out.", "Earlier the customer said the blender arrived broken and they want their money returned.", "Earlier the customer said they mostly shop on weekends after work.", ["money returned"]),
    ("cancel_order", "Please stop it.", "Earlier the customer said the pair of running shoes was bought by mistake.", "Earlier the customer said they are a student at a local college.", ["by mistake"]),
    ("change_order", "Can that be different?", "Earlier the customer said the shirt in their basket should be size medium, not large.", "Earlier the customer said they enjoy hiking with friends on weekends.", ["size medium"]),
    ("change_shipping_address", "It needs to go somewhere else.", "Earlier the customer said they are moving to a new apartment across town next week.", "Earlier the customer said their favourite colour is green and blue.", ["moving to a new apartment"]),
    ("set_up_shipping_address", "I haven't given you that yet.", "Earlier the customer said this is their first purchase and no drop-off location is saved.", "Earlier the customer said they usually speak Portuguese when at home.", ["no drop-off location"]),
    ("delivery_period", "Will it make it in time?", "Earlier the customer said the gift is needed for a birthday party on Friday.", "Earlier the customer said they have two cats and a small garden.", ["birthday party on friday"]),
    ("delivery_options", "What are my choices?", "Earlier the customer said the parcel could go to a pickup point or straight to their door.", "Earlier the customer said they are learning to play the guitar this year.", ["pickup point"]),
    ("check_cancellation_fee", "What would that cost me?", "Earlier the customer said they are thinking about ending their premium plan early.", "Earlier the customer said they mostly read the news on their tablet.", ["ending their premium plan early"]),
    ("check_refund_policy", "How does that work here?", "Earlier the customer said they are unsure whether opened items can be sent back for money.", "Earlier the customer said they take the train to work most mornings.", ["opened items"]),
    ("check_payment_methods", "Which of those do you take?", "Earlier the customer said they only own a debit card and a digital wallet.", "Earlier the customer said they will be travelling abroad next summer.", ["digital wallet"]),
    ("check_invoice", "Can I look at it?", "Earlier the customer said their accountant wants to see the VAT amount on last month's bill.", "Earlier the customer said they have lived in the same town for years.", ["vat amount"]),
    ("get_invoice", "Please send me that.", "Earlier the customer said their accountant needs a copy of the bill from last month.", "Earlier the customer said they are planning a small dinner for friends.", ["copy of the bill"]),
    ("complaint", "This is unacceptable.", "Earlier the customer said the driver was rude and left the parcel out in the rain.", "Earlier the customer said they were just chatting about the weather today.", ["driver was rude"]),
    ("contact_customer_service", "How do I reach you?", "Earlier the customer said they want the opening hours and phone number of the help desk.", "Earlier the customer said they like listening to jazz while working.", ["phone number"]),
    ("contact_human_agent", "Let's take this offline.", "Earlier the customer said the automated replies keep misunderstanding them and they want a real person.", "Earlier the customer said they usually read reviews of restaurants before going out.", ["real person"]),
    ("create_account", "Yes, let's do that.", "Earlier the customer said they have only shopped as a guest and now want their own login.", "Earlier the customer said they are training for a half marathon in the spring.", ["own login"]),
    ("delete_account", "Please wipe it all.", "Earlier the customer said they are leaving the service for good and want their personal data removed.", "Earlier the customer said they grow tomatoes on their balcony each year.", ["personal data removed"]),
    ("edit_account", "It's out of date.", "Earlier the customer said their phone number and surname are different since getting married.", "Earlier the customer said they are fond of old science fiction novels.", ["surname"]),
    ("newsletter_subscription", "Stop those, actually.", "Earlier the customer said the weekly promotional emails keep filling up their inbox.", "Earlier the customer said they bake bread every Sunday morning.", ["promotional emails"]),
    ("recover_password", "I can't get in.", "Earlier the customer said they have forgotten the secret phrase for their login.", "Earlier the customer said they usually go to bed quite early each night.", ["secret phrase"]),
    ("registration_problems", "It keeps failing at the last step.", "Earlier the customer said the form keeps rejecting the details they type in to join.", "Earlier the customer said they watched a documentary about the ocean.", ["keeps rejecting"]),
    ("review", "I want to say something about it.", "Earlier the customer said the headphones sound great and deserve five stars.", "Earlier the customer said they take a long walk around the park every evening.", ["five stars"]),
    ("switch_account", "Use the other one.", "Earlier the customer said they hold a work profile and a personal profile and are in the wrong one.", "Earlier the customer said they are saving up for a trip to the mountains.", ["work profile"]),
    ("place_order", "Yes, I'll take it.", "Earlier the customer said they compared two laptops and have now decided on the 14-inch one.", "Earlier the customer said they keep a small notebook of quotes at their desk.", ["decided on the 14-inch"]),
    ("track_order", "Where are we on that?", "Earlier the customer said the parcel left the warehouse on Tuesday and the courier link shows nothing.", "Earlier the customer said they have started learning to paint with watercolours.", ["courier link"]),
    ("get_refund", "Make it right.", "Earlier the customer said the concert tickets were charged twice and they want one charge returned.", "Earlier the customer said they have a younger brother who studies architecture.", ["charged twice"]),
    ("payment_issue", "It says error.", "Earlier the customer said the card was charged but the confirmation page never appeared.", "Earlier the customer said they adopted a rescue dog from a local shelter.", ["confirmation page never appeared"]),
]


def candidates_by_intent() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in load_sample("main"):
        out.setdefault(r.ground_truth, list(r.choices))
    return out


def _has_keyword(text: str, intents: list[str]) -> list[str]:
    t = f" {text.lower()} "
    return [f"{i}:{k}" for i in intents for k in KEYWORDS.get(i, []) if k in t]


def validate_triplet(label: str, target: str, relevant: str, irrelevant: str, cues: list[str], candidates: list[str]) -> list[str]:
    """Problems that would make this triplet's relevance claim unsound. Empty list = passes every check."""
    problems: list[str] = []
    if label not in candidates:
        problems.append("expected label not among candidates")
    if not cues:
        problems.append("no relevance cues")
    rel, irr, tgt = relevant.lower(), irrelevant.lower(), target.lower()
    for cue in cues:
        c = cue.lower()
        if c not in rel:
            problems.append(f"cue {cue!r} missing from relevant history")
        if c in irr:
            problems.append(f"cue {cue!r} appears in irrelevant history")
        if c in tgt:
            problems.append(f"cue {cue!r} appears in the target message")
    others = [c for c in candidates if c != label and KEYWORDS.get(c) != KEYWORDS.get(label)]
    if hits := _has_keyword(relevant, others):
        problems.append(f"relevant history names other candidates: {hits}")
    if hits := _has_keyword(irrelevant, candidates):
        problems.append(f"irrelevant history names a candidate intent: {hits}")
    if hits := _has_keyword(target, candidates):
        problems.append(f"target message is not vague, it names: {hits}")
    lo, hi = LENGTH_RATIO_RANGE
    ratio = len(irrelevant) / len(relevant)
    if not lo <= ratio <= hi:
        problems.append(f"history lengths not comparable (irrelevant/relevant = {ratio:.2f})")
    return problems


def rows_for_triplet(index: int, label: str, target: str, relevant: str, irrelevant: str, cues: list[str], candidates: list[str]) -> list[dict[str, Any]]:
    tid = f"triplet-{index:03d}"
    problems = validate_triplet(label, target, relevant, irrelevant, cues, candidates)
    if problems:
        raise StressError(f"{tid} failed validation: {problems}")
    histories = {"no_context": [], "irrelevant_context": [irrelevant], "relevant_context": [relevant]}
    rows = []
    for cond in CONDITIONS:
        payload = {"message": target} if not histories[cond] else {"message": target, "history": histories[cond]}
        rows.append(
            {
                "variant_id": f"{tid}:{cond}",
                "source_example_id": tid,
                "context_condition": cond,
                "text": target,
                "base_message": target,
                "state_payload": payload,
                "history": histories[cond],
                "relevant_history": [relevant],
                "irrelevant_history": [irrelevant],
                "relevance_cues": cues,
                "expected_label": label,
                "ground_truth": label,
                "candidates": candidates,
                "intent": label,
                "relevance_basis": "hand-authored; validated by cue/keyword checks; pending human review",
            }
        )
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cand = candidates_by_intent()
    rows: list[dict[str, Any]] = []
    for i, (label, target, relevant, irrelevant, cues) in enumerate(TRIPLETS, start=1):
        rows.extend(rows_for_triplet(i, label, target, relevant, irrelevant, cues, cand[label]))
    meta = {
        "experiment": NAME,
        **seed_fields(None, "Hand-authored triplets with no randomness or sampling; the Bitext main sample is only read to look up each intent's candidate list."),
        "n_triplets": len(TRIPLETS),
        "conditions": list(CONDITIONS),
        "generation_method": "hand-authored triplets written once by the project author (no model generation); candidates are the real Bitext choice set for the expected intent",
        "relevance_establishment": "vague target + cue phrases only in the relevant history + keyword validator (see module docstring); not proof, hence human review",
        "state_shape": {"no_context": {"message": "<target>"}, "others": {"message": "<target>", "history": ["<one history sentence>"]}},
        **REVIEW_FIELDS,
    }
    return rows, meta


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    per = df[df["context_condition"] == "no_context"]
    triplets = [
        {"triplet": r["source_example_id"], "expected_label": r["expected_label"], "target": r["base_message"], "relevant": r["relevant_history"][0], "irrelevant": r["irrelevant_history"][0]}
        for _, r in per.head(12).iterrows()
    ]
    return {
        "n_triplets": df["source_example_id"].nunique(),
        "n_rows": len(df),
        "counts_by_condition": df["context_condition"].value_counts().to_dict(),
        "distinct_expected_intents": int(per["expected_label"].nunique()),
        "example_triplets": triplets,
    }


TRANSITIONS = ("wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong")


def transition_counts(paired: pd.DataFrame) -> dict[str, int]:
    """Counts of base->variant correctness transitions among pairs where both sides were scored."""
    ok = paired[paired["correct_base"].notna() & paired["correct_variant"].notna()]
    b, v = ok["correct_base"].astype(bool), ok["correct_variant"].astype(bool)
    return {
        "wrong_to_correct": int((~b & v).sum()),
        "correct_to_wrong": int((b & ~v).sum()),
        "correct_to_correct": int((b & v).sum()),
        "wrong_to_wrong": int((~b & ~v).sum()),
        "n_scored_pairs": len(ok),
    }


def analyze(results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not (results["context_condition"] == "no_context").any():
        raise StressError("No 'no_context' rows; paired comparison needs each triplet's no-context row.")
    base = results[results["context_condition"] == "no_context"]
    rows, trans_rows, pair_rows = [], [], []
    for provider, prov in results.groupby("provider"):
        for cond in CONDITIONS:
            grp = prov[prov["context_condition"] == cond]
            ok = grp[grp["prediction"].notna()]
            rows.append(
                {
                    "provider": provider,
                    "context_condition": cond,
                    "n": len(grp),
                    "n_scored": int(ok["correct"].notna().sum()),
                    "accuracy": float(ok["correct"].dropna().astype(bool).mean()) if ok["correct"].notna().any() else None,
                    "mean_confidence": float(ok["confidence"].mean()) if len(ok) else None,
                }
            )
        b = base[base["provider"] == provider]
        for cond in ("irrelevant_context", "relevant_context"):
            paired = pair_results(b, prov[prov["context_condition"] == cond])
            m = paired_metrics(paired)
            rows_idx = next(r for r in rows if r["provider"] == provider and r["context_condition"] == cond)
            rows_idx.update({"n_pairs_vs_no_context": m["n_pairs"], "decision_flips_vs_no_context": m.get("decision_flip_rate"), "accuracy_change_vs_no_context": m.get("accuracy_delta"), "mean_confidence_delta_vs_no_context": m.get("mean_confidence_delta")})
            trans_rows.append({"provider": provider, "comparison": f"{cond} vs no_context", **transition_counts(paired)})
            for _, p in paired.iterrows():
                pair_rows.append({"provider": provider, "source_example_id": p["source_example_id"], "context_condition": cond, "expected_label": p["expected_label"], "no_context_prediction": p["prediction_base"], "prediction": p["prediction_variant"], "no_context_correct": p["correct_base"], "correct": p["correct_variant"], "no_context_confidence": p["confidence_base"], "confidence": p["confidence_variant"]})
        direct = pair_results(prov[prov["context_condition"] == "irrelevant_context"], prov[prov["context_condition"] == "relevant_context"])
        trans_rows.append({"provider": provider, "comparison": "relevant_context vs irrelevant_context", **transition_counts(direct)})
    summary = {
        "experiment": NAME,
        "n_results": len(results),
        "pairing": "every condition is compared with the same triplet's no_context row (same provider); relevant vs irrelevant is also compared directly",
        "transition_definition": "from the base condition to the compared condition: wrong_to_correct means the base answer was wrong and the compared answer is correct",
        "interpretation": "not concluded here; read transitions and accuracy with the n shown",
    }
    return {"relevance_by_condition": pd.DataFrame(rows), "relevance_transitions": pd.DataFrame(trans_rows), "relevance_pairs": pd.DataFrame(pair_rows)}, summary


SPEC = ExperimentSpec(name=NAME, dataset_dir=DATASET_DIR, results_dir=RESULTS_DIR, build=build, describe=describe, analyze=analyze)
