"""Phase II Step 6: deterministic text-noise robustness.

Question: how robust is Jev when the meaning stays approximately the same but the text gets messy?

Eight noise types x severity 1 (light) / 2 (moderate) / 3 (heavy), plus a severity-0 clean control
per source. Every transformation is a pure function of (source text, source_example_id, seed,
noise_type, severity): the same inputs always give the same output, no model is involved, and the
frozen dataset stores every transformed text.

Protected spans: Bitext placeholders (`{{...}}`) are swapped out before any transformation runs and restored
byte-for-byte afterwards, so no noise type (typos, casing, punctuation, whitespace, emoji, abbreviations,
duplication) can alter them.

Semantic integrity: severe noise can make text unreadable or change its meaning, and that must not
be silently counted as a model failure. Each row gets `semantic_integrity` = valid | questionable,
from a deterministic heuristic (character similarity and token recall against the original, with
emoji ignored). It is a FLAG for reporting, not a human judgement. The PRIMARY robustness summaries use
rows marked valid; every table that splits by integrity, an all-rows sensitivity table and the counts of
questionable rows are also written, and no questionable row is ever discarded.
"""

import difflib
import re
import string
from typing import Any

import pandas as pd

from phase2.stress import (
    PHASE2_RESULTS_DIR,
    REVIEW_FIELDS,
    STRESS_DIR,
    ExperimentSpec,
    StressError,
    grouped_paired_metrics,
    pair_results,
    protected,
    seeded_rng,
    select_bitext_sources,
)

NAME = "noise"
DATASET_DIR = STRESS_DIR / "noise"
RESULTS_DIR = PHASE2_RESULTS_DIR / "noise"
N_SOURCES = 100
NOISE_SEED = 6
SEVERITIES = (1, 2, 3)
EMOJIS = ["🙂", "😡", "🙏", "😅", "👍", "❗"]
KEY_NEIGHBORS = dict(zip("qwertyuiopasdfghjklzxcvbnm", "wertyuiopqsdfghjklaxcvbnmz", strict=True))
ABBREVIATIONS = {"please": "pls", "you": "u", "your": "ur", "are": "r", "thanks": "thx", "because": "bc", "information": "info", "account": "acct", "message": "msg", "number": "no.", "customer": "cust", "service": "svc", "could": "cud", "about": "abt", "tomorrow": "tmrw", "payment": "pmt", "delivery": "deliv", "with": "w/", "for": "4", "to": "2", "and": "&"}
PROTECTED = "{}"  # braces are never punctuation-noise targets; whole {{...}} spans are additionally protected by phase2.stress.protected
INTEGRITY_MIN_CHAR_RATIO = 0.70
INTEGRITY_MIN_TOKEN_RECALL = 0.50


def _sev(sev: int, mapping: dict[int, Any]) -> Any:
    if sev not in mapping:
        raise StressError(f"severity must be one of {sorted(mapping)}, got {sev}")
    return mapping[sev]


def typos(text: str, rng, sev: int) -> str:
    chars = list(text)
    for _ in range(_sev(sev, {1: 1, 2: 3, 3: 6})):
        idxs = [i for i, c in enumerate(chars) if c.isalpha()]
        if len(idxs) < 3:
            break
        i = rng.choice(idxs[1:-1])
        op = rng.choice(["swap", "delete", "dup", "sub"])
        if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "delete":
            del chars[i]
        elif op == "dup":
            chars.insert(i, chars[i])
        else:
            chars[i] = KEY_NEIGHBORS.get(chars[i].lower(), chars[i])
    return "".join(chars)


def missing_punctuation(text: str, rng, sev: int) -> str:
    punct = "".join(c for c in string.punctuation if c not in PROTECTED)
    _sev(sev, {1: 0, 2: 0, 3: 0})
    if sev == 1:
        return text.rstrip(punct + " ")
    positions = [i for i, c in enumerate(text) if c in punct]
    drop = set(positions if sev == 3 else positions[::2])
    out = "".join(c for i, c in enumerate(text) if i not in drop)
    return out.rstrip(punct + " ") if sev == 3 else out


def extra_punctuation(text: str, rng, sev: int) -> str:
    words = text.split(" ")
    _sev(sev, {1: 0, 2: 0, 3: 0})
    if sev == 1:
        return text.rstrip() + rng.choice([".", "?", "!"])
    if sev == 2:
        if len(words) > 1:
            words[0] += ","
        return " ".join(words).rstrip() + "!!"
    return " ".join(w + ("..." if i % 3 == 2 else "") for i, w in enumerate(words)).rstrip() + "!!!???"


def capitalization(text: str, rng, sev: int) -> str:
    _sev(sev, {1: 0, 2: 0, 3: 0})
    if sev == 1:
        return text[:1].swapcase() + text[1:] if text else text
    if sev == 2:
        return " ".join(w.upper() if rng.random() < 0.3 else w for w in text.split(" "))
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)


def whitespace(text: str, rng, sev: int) -> str:
    _sev(sev, {1: 0, 2: 0, 3: 0})
    words = text.split(" ")
    gaps = list(range(len(words) - 1))
    if not gaps:
        return text
    count = {1: 1, 2: 3, 3: 4}[sev]
    for g in rng.sample(gaps, min(count, len(gaps))):
        words[g] += " " * (1 if sev == 1 else rng.randint(1, 3))
    out = " ".join(words)
    if sev >= 2:
        out = "  " + out + "   "
    if sev == 3:
        long_words = [m for m in re.finditer(r"[A-Za-z]{6,}", out)]
        if long_words:
            m = rng.choice(long_words)
            mid = m.start() + len(m.group()) // 2
            out = out[:mid] + " " + out[mid:]
        out = out.replace(" ", "\t", 1)
    return out


def emoji(text: str, rng, sev: int) -> str:
    _sev(sev, {1: 0, 2: 0, 3: 0})
    if sev == 1:
        return f"{text} {rng.choice(EMOJIS)}"
    words = text.split(" ")
    if sev == 2:
        if len(words) > 2:
            words.insert(len(words) // 2, rng.choice(EMOJIS))
        return " ".join(words) + " " + "".join(rng.choice(EMOJIS) for _ in range(2))
    out: list[str] = []
    for i, w in enumerate(words):
        out.append(w)
        if i % 3 == 2:
            out.append(rng.choice(EMOJIS))
    return " ".join(out) + " " + "".join(rng.choice(EMOJIS) for _ in range(3))


def abbreviations(text: str, rng, sev: int) -> str:
    limit = _sev(sev, {1: 1, 2: 3, 3: 10**6})
    done = 0

    def swap(m: re.Match) -> str:
        nonlocal done
        if done >= limit:
            return m.group(0)
        done += 1
        return ABBREVIATIONS[m.group(0).lower()]

    return re.sub(r"\b(" + "|".join(sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b", swap, text, flags=re.IGNORECASE)


def word_duplication(text: str, rng, sev: int) -> str:
    words = text.split(" ")
    if not words:
        return text
    n = {1: 1, 2: 3, 3: max(6, len(words) // 2)}[_sev(sev, {1: 1, 2: 3, 3: 6}) and sev]
    for i in sorted(rng.sample(range(len(words)), min(n, len(words))), reverse=True):
        words.insert(i, words[i])
    return " ".join(words)


NOISE_FUNCS = {
    "typos": typos,
    "missing_punctuation": missing_punctuation,
    "extra_punctuation": extra_punctuation,
    "capitalization": capitalization,
    "whitespace": whitespace,
    "emoji": emoji,
    "abbreviations": abbreviations,
    "word_duplication": word_duplication,
}


def apply_noise(text: str, source_id: str, noise_type: str, severity: int, seed: int = NOISE_SEED) -> str:
    """Severity 0 returns the text unchanged. Otherwise a pure function of all arguments."""
    if severity == 0:
        return text
    if noise_type not in NOISE_FUNCS:
        raise StressError(f"unknown noise_type {noise_type!r}")
    return protected(NOISE_FUNCS[noise_type])(text, seeded_rng(source_id, seed, noise_type, severity), severity)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"[^\x00-\x7f]", " ", text.lower()))


def semantic_integrity(original: str, transformed: str) -> tuple[str, float, float]:
    """(valid|questionable, char_ratio, token_recall). Emoji/non-ASCII are ignored; a heuristic flag only."""
    a, b = "".join(_tokens(original)), "".join(_tokens(transformed))
    ratio = difflib.SequenceMatcher(None, a, b).ratio() if (a or b) else 1.0
    orig_tokens, new_tokens = _tokens(original), set(_tokens(transformed))
    recall = sum(t in new_tokens for t in orig_tokens) / len(orig_tokens) if orig_tokens else 1.0
    ok = ratio >= INTEGRITY_MIN_CHAR_RATIO and recall >= INTEGRITY_MIN_TOKEN_RECALL
    return ("valid" if ok else "questionable"), round(ratio, 4), round(recall, 4)


def rows_for_source(source_id: str, text: str, choices: list[str], truth: str, seed: int = NOISE_SEED) -> tuple[list[dict[str, Any]], int]:
    """Clean control row + every noise variant with new text. Returns (rows, n_skipped_as_unchanged_or_duplicate)."""
    base = {"source_example_id": source_id, "original_text": text, "expected_label": truth, "ground_truth": truth, "candidates": choices, "intent": truth, "seed": seed}
    rows = [{**base, "variant_id": f"{source_id}:clean", "noise_type": "none", "severity": 0, "text": text, "transformed_text": text, "semantic_integrity": "valid", "integrity_char_ratio": 1.0, "integrity_token_recall": 1.0}]
    skipped = 0
    seen = {text}
    for noise_type in NOISE_FUNCS:
        for sev in SEVERITIES:
            out = apply_noise(text, source_id, noise_type, sev, seed)
            if out in seen:  # unchanged, or identical to a variant already produced for this source
                skipped += 1
                continue
            seen.add(out)
            verdict, ratio, recall = semantic_integrity(text, out)
            rows.append({**base, "variant_id": f"{source_id}:{noise_type}:s{sev}", "noise_type": noise_type, "severity": sev, "text": out, "transformed_text": out, "semantic_integrity": verdict, "integrity_char_ratio": ratio, "integrity_token_recall": recall})
    return rows, skipped


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for r in sorted(select_bitext_sources(N_SOURCES), key=lambda r: r.id):
        src_rows, s = rows_for_source(r.id, r.text, r.choices, r.ground_truth)
        rows.extend(src_rows)
        skipped += s
    meta = {
        "experiment": NAME,
        "source": "data/samples/bitext_main.jsonl via phase2.stress.select_bitext_sources",
        "n_sources_requested": N_SOURCES,
        "seed": NOISE_SEED,
        "noise_types": list(NOISE_FUNCS),
        "severities": [0, *SEVERITIES],
        "generation_method": "deterministic pure functions of (text, source_example_id, seed, noise_type, severity); no model",
        "semantic_integrity_rule": f"questionable if char_ratio < {INTEGRITY_MIN_CHAR_RATIO} or token_recall < {INTEGRITY_MIN_TOKEN_RECALL} (heuristic flag, not human-verified)",
        "variants_skipped_because_text_unchanged_or_duplicate": skipped,
        **REVIEW_FIELDS,
    }
    return rows, meta


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    noisy = df[df["severity"] > 0]
    sample = []
    for nt in NOISE_FUNCS:
        for sev in SEVERITIES:
            hit = noisy[(noisy["noise_type"] == nt) & (noisy["severity"] == sev)].head(1)
            if len(hit):
                r = hit.iloc[0]
                sample.append({"noise_type": nt, "severity": sev, "original": r["original_text"], "transformed": r["text"], "semantic_integrity": r["semantic_integrity"]})
    return {
        "n_source_examples": df["source_example_id"].nunique(),
        "n_rows_including_clean": len(df),
        "n_noisy_variants": len(noisy),
        "counts_by_noise_type": noisy["noise_type"].value_counts().to_dict(),
        "counts_by_severity": {int(k): int(v) for k, v in noisy["severity"].value_counts().sort_index().items()},
        "semantic_integrity_counts": noisy["semantic_integrity"].value_counts().to_dict(),
        "questionable_by_severity": noisy[noisy["semantic_integrity"] == "questionable"]["severity"].value_counts().sort_index().to_dict(),
        "examples": sample[::3][:8],
    }


def _with_drop(table: pd.DataFrame) -> pd.DataFrame:
    if "accuracy_delta" in table:
        table.insert(table.columns.get_loc("accuracy_delta") + 1, "accuracy_drop", -table["accuracy_delta"])
    return table


def questionable_counts(paired: pd.DataFrame) -> pd.DataFrame:
    """Per provider and severity: how many pairs are valid vs questionable (nothing is dropped)."""
    counts = paired.groupby(["provider", "severity", "semantic_integrity"]).size().unstack("semantic_integrity", fill_value=0)
    for col in ("valid", "questionable"):
        if col not in counts:
            counts[col] = 0
    counts["n_pairs"] = counts["valid"] + counts["questionable"]
    counts["questionable_share"] = counts["questionable"] / counts["n_pairs"]
    return counts[["valid", "questionable", "n_pairs", "questionable_share"]].reset_index()


def analyze(results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not (results["severity"] == 0).any():
        raise StressError("No severity-0 (clean) rows; paired comparison needs each source's clean row.")
    base = results[results["severity"] == 0]
    noisy = results[results["severity"] > 0]
    paired = pair_results(base, noisy).assign(dataset="bitext")
    valid_only = paired[paired["semantic_integrity"] == "valid"]
    tables = {
        "noise_by_severity": _with_drop(grouped_paired_metrics(paired, ["provider", "severity", "semantic_integrity"])),
        "noise_by_type": _with_drop(grouped_paired_metrics(paired, ["provider", "noise_type", "semantic_integrity"])),
        "noise_by_type_severity": _with_drop(grouped_paired_metrics(paired, ["provider", "noise_type", "severity", "semantic_integrity"])),
        "noise_valid_only_by_severity": _with_drop(grouped_paired_metrics(valid_only, ["provider", "severity"])),  # PRIMARY summary
        "noise_all_rows_by_severity": _with_drop(grouped_paired_metrics(paired, ["provider", "severity"])),  # sensitivity: valid + questionable
        "noise_questionable_counts": questionable_counts(paired),
        "noise_by_intent": _with_drop(grouped_paired_metrics(valid_only, ["provider", "dataset", "intent"])),
        "noise_pairs": paired[["provider", "source_example_id", "noise_type", "severity", "semantic_integrity", "expected_label", "prediction_base", "prediction_variant", "confidence_base", "confidence_variant", "correct_base", "correct_variant"]],
    }
    summary = {
        "experiment": NAME,
        "n_results": len(results),
        "n_pairs": len(paired),
        "n_pairs_semantic_integrity_valid": len(valid_only),
        "n_pairs_semantic_integrity_questionable": len(paired) - len(valid_only),
        "primary_summary": "noise_valid_only_by_severity (semantic_integrity == valid)",
        "sensitivity_summary": "noise_all_rows_by_severity (all rows, valid + questionable); noise_questionable_counts lists the questionable rows per severity",
        "pairing": "each noisy variant vs its own source's severity-0 row (same provider)",
        "note": "questionable rows are kept, reported separately and never silently discarded or counted as model failures",
        "interpretation": "not concluded here; read rates together with n_pairs",
    }
    return tables, summary


SPEC = ExperimentSpec(name=NAME, dataset_dir=DATASET_DIR, results_dir=RESULTS_DIR, build=build, describe=describe, analyze=analyze)
