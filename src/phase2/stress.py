"""Shared infrastructure for the Phase II stress experiments (Steps 4-11).

Every stress experiment follows the same shape:

    build   -> deterministic FROZEN dataset under data/stress/<name>/ (never regenerated at eval time)
    report  -> dry-run: counts and expected API calls, constructs no client, makes no call
    run     -> live evaluation; refuses unless --approve-calls equals the exact expected call count
    analyze -> offline analysis of already-recorded results

Nothing in this module makes a model call unless `execute_live` is invoked with an explicit,
matching call-count approval. Frozen datasets and result files are never overwritten.

Row contract for a frozen dataset (JSONL, one object per line):
    variant_id         unique id of this evaluated input (becomes PredictionResult.example_id)
    source_example_id  id of the original example this variant derives from (always retained)
    text               what is sent as the state message
    candidates         the allowed choices
    ground_truth       single expected label, or null when no single label is defensible
    state_payload      optional: a structured state (e.g. {"message", "history"}) sent instead of {"user_message": text}
    ...                any experiment-specific fields (kept verbatim and joined onto results)
"""

import asyncio
import hashlib
import json
import random
import re
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from typesafe_sdk import Choice, JSONContent, TypeSafeError

from dataset_loaders.bitext import BitextRow, load_sample
from evaluation.pricing import estimate_cost_usd
from evaluation.recorder import results_to_dataframe
from evaluation.runner import run_evaluation_async
from models.experiment import ExperimentConfig
from models.prediction import Example, PredictionResult

STRESS_DIR = Path("data/stress")
PHASE2_RESULTS_DIR = Path("data/results/phase2")
SCHEMA_VERSION = 1
SOURCE_SEED = 20260924  # fixed seed for choosing source examples shared by the stress experiments
REQUIRED_ROW_FIELDS = ["variant_id", "source_example_id", "text", "candidates"]
QUESTION_INSTRUCTIONS = "Select the intent that best represents the user's request."


class StressError(Exception):
    """A stress dataset or run could not be built, loaded or approved."""


# --- determinism helpers -------------------------------------------------------------------------


def stable_int(*parts: object) -> int:
    """Deterministic integer from arbitrary parts; independent of PYTHONHASHSEED and process."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def seeded_rng(*parts: object) -> random.Random:
    return random.Random(stable_int(*parts))


REVIEW_FIELDS = {
    "review_status": "approved",
    "review_note": "Datasets were reviewed by the project owner before any live run (methodology cleanup and spot-check packets, 2026-09-25). Approval was recorded in this manifest after the live runs completed; no dataset row or hash changed.",
    "review_recorded_on": "2026-09-25",
}


def seed_fields(seed: int | None, note: str) -> dict[str, Any]:
    """Manifest fields describing a dataset's randomness. `seed=None` means none is involved (not_applicable), never "unknown"."""
    return {"seed": seed, "seed_status": "recorded" if seed is not None else "not_applicable", "seed_note": note}


def source_selection_seed_fields(derivation: str) -> dict[str, Any]:
    """For datasets whose source examples come from `select_bitext_sources` (seeded by SOURCE_SEED)."""
    return seed_fields(SOURCE_SEED, f"SOURCE_SEED seeds the selection of source examples (select_bitext_sources). {derivation}")


# --- protected spans -----------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")  # Bitext placeholders such as {{Order Number}}
_SENTINEL_BASE = 0xE000  # Unicode private-use characters: not alphabetic, not punctuation, unchanged by case folding


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def protected(fn: Callable[..., str]) -> Callable[..., str]:
    """Wrap a text transformation `fn(text, ...)` so `{{...}}` spans pass through byte-for-byte unchanged.

    Each placeholder is swapped for one private-use character before the transformation runs and swapped
    back afterwards, so case, typo, punctuation, whitespace and abbreviation transforms cannot touch it.
    """

    def wrapper(text: str, *args: Any, **kwargs: Any) -> str:
        spans = placeholders(text)
        if not spans:
            return fn(text, *args, **kwargs)
        if any(_SENTINEL_BASE <= ord(c) < _SENTINEL_BASE + len(spans) for c in text):
            raise StressError("text already contains the reserved private-use characters used to protect placeholders")
        masked, pos = [], 0
        for i, m in enumerate(PLACEHOLDER_RE.finditer(text)):
            masked.append(text[pos : m.start()] + chr(_SENTINEL_BASE + i))
            pos = m.end()
        masked.append(text[pos:])
        out = fn("".join(masked), *args, **kwargs)
        for i, span in enumerate(spans):
            out = out.replace(chr(_SENTINEL_BASE + i), span)
        return out

    wrapper.__name__ = getattr(fn, "__name__", "protected")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def select_bitext_sources(n: int = 100, seed: int = SOURCE_SEED, sample: str = "main") -> list[BitextRow]:
    """A fixed, seeded subset of the frozen Bitext main sample, shared across stress experiments."""
    rows = sorted(load_sample(sample), key=lambda r: r.id)
    return seeded_rng("bitext-sources", seed).sample(rows, n)


# --- frozen datasets -----------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Every row carries the contract fields, `variant_id` is unique, and `source_example_id` is non-empty."""
    seen: set[str] = set()
    for row in rows:
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row or row[f] in (None, "")]
        if missing:
            raise StressError(f"row {row.get('variant_id')!r} is missing required field(s): {missing}")
        if row["variant_id"] in seen:
            raise StressError(f"duplicate variant_id {row['variant_id']!r}")
        seen.add(row["variant_id"])
        if not row["candidates"]:
            raise StressError(f"row {row['variant_id']!r} has no candidates")


def write_frozen_dataset(directory: Path, rows: Sequence[dict[str, Any]], meta: dict[str, Any], name: str = "dataset") -> dict[str, Any]:
    """Write `<name>.jsonl` + `<name>_manifest.json` deterministically. Refuses to overwrite; files become read-only."""
    validate_rows(rows)
    data_path, manifest_path = directory / f"{name}.jsonl", directory / f"{name}_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise StressError(f"Frozen dataset already exists in {directory}; refusing to overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": name,
        "n_rows": len(rows),
        "n_source_examples": len({r["source_example_id"] for r in rows}),
        "sha256": _sha256_bytes(payload),
        **meta,
    }
    data_path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    for p in (data_path, manifest_path):
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return manifest


def load_frozen_dataset(directory: Path, name: str = "dataset") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load rows + manifest, verifying the content hash so a silently edited dataset is detected."""
    data_path, manifest_path = directory / f"{name}.jsonl", directory / f"{name}_manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        raise StressError(f"No frozen dataset '{name}' in {directory}; run the build step first.")
    manifest = json.loads(manifest_path.read_text())
    raw = data_path.read_bytes()
    if _sha256_bytes(raw) != manifest["sha256"]:
        raise StressError(f"{data_path} does not match its manifest hash; the frozen dataset was modified.")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    validate_rows(rows)
    return rows, manifest


# --- evaluation ----------------------------------------------------------------------------------


def row_to_example(row: dict[str, Any], dataset: str) -> Example:
    return Example(
        example_id=row["variant_id"],
        dataset=dataset,
        state=row["text"],
        candidates=list(row["candidates"]),
        ground_truth=row.get("ground_truth"),
        locale=row.get("locale"),
    )


def default_state(example: Example) -> JSONContent:
    return {"user_message": example.state}


def top_margin(probabilities: dict[str, float] | None) -> float | None:
    """top-1 minus top-2 probability. None unless a distribution over >= 2 choices was actually recorded."""
    if not probabilities or len(probabilities) < 2:
        return None
    top = sorted(probabilities.values(), reverse=True)
    return float(top[0] - top[1])


class StressEvaluator:
    """Like `SystemOneEvaluator`, but with a configurable state and it keeps the full probabilities the
    SDK returns (Phase I's evaluator stores only the chosen label's probability).

    `state_builder(example)` lets an experiment send a structured state (e.g. message + history).
    """

    def __init__(
        self,
        client: Any,
        experiment: str,
        provider: str,
        state_builder: Callable[[Example], JSONContent] = default_state,
    ) -> None:
        self._client = client
        self._experiment = experiment
        self.provider = provider
        self._state_builder = state_builder
        self.probabilities: dict[str, dict[str, float]] = {}

    async def predict(self, example: Example) -> PredictionResult:
        start = time.perf_counter()
        base = dict(
            experiment=self._experiment,
            provider=self.provider,
            example_id=example.example_id,
            dataset=example.dataset,
            locale=example.locale,
            ground_truth=example.ground_truth,
            candidates=example.candidates,
            text=example.state,
        )
        try:
            response = await self._client.system_one(
                state=self._state_builder(example),
                questions={"intent": Choice(instructions=QUESTION_INSTRUCTIONS, criteria={c: None for c in example.candidates})},
            )
        except TypeSafeError as exc:
            return PredictionResult(
                **base, prediction=None, correct=None, confidence=None, latency_ms=(time.perf_counter() - start) * 1000, error=str(exc), request_id=getattr(exc, "request_id", None)
            )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices["intent"]
        prediction = answer.choice
        self.probabilities[example.example_id] = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None
        usage = response.usage
        in_total = getattr(usage, "input_tokens_total", None)
        out_total = getattr(usage, "output_tokens_total", None)
        input_tokens = in_total if in_total is not None else usage.input_tokens
        output_tokens = out_total if out_total is not None else usage.output_tokens
        return PredictionResult(
            **base,
            prediction=prediction,
            correct=(prediction == example.ground_truth) if example.ground_truth is not None else None,
            confidence=answer.probabilities.get(prediction),
            latency_ms=latency_ms,
            error=None,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost_usd(response.model, input_tokens, output_tokens),
        )


def results_frame(results: Sequence[PredictionResult], rows: Sequence[dict[str, Any]], probabilities: dict[str, dict[str, float]], config: ExperimentConfig) -> pd.DataFrame:
    """PredictionResult rows joined with the frozen dataset's fields (never dropping source_example_id)."""
    df = results_to_dataframe(list(results), config)
    meta = pd.DataFrame(list(rows)).drop(columns=["text", "candidates", "ground_truth", "locale"], errors="ignore")
    for col in meta.columns:  # list/dict cells are stored as JSON so the frame stays parquet-friendly
        meta[col] = meta[col].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v)
    df = df.merge(meta, left_on="example_id", right_on="variant_id", how="left", validate="one_to_one")
    df["probabilities_json"] = df["example_id"].map(lambda i: json.dumps(probabilities[i], sort_keys=True) if i in probabilities else None)
    return df


@dataclass
class CallPlan:
    """What a live run would cost in API calls. Produced without constructing any client."""

    experiment: str
    n_rows: int
    providers: list[str]
    calls_per_provider: int
    total_calls: int
    extra: dict[str, Any] = field(default_factory=dict)


def plan_calls(experiment: str, rows: Sequence[dict[str, Any]], providers: Sequence[str], **extra: Any) -> CallPlan:
    return CallPlan(experiment, len(rows), list(providers), len(rows), len(rows) * len(providers), dict(extra))


def new_output_path(directory: Path, provider: str) -> Path:
    path = directory / f"raw_results_{provider.replace(':', '_').replace('/', '_')}.parquet"
    if path.exists():
        raise StressError(f"{path} already exists; refusing to overwrite recorded results.")
    return path


async def execute_live(
    experiment: str,
    rows: Sequence[dict[str, Any]],
    dataset: str,
    provider_specs: Sequence[str],
    results_dir: Path,
    approve_calls: int | None,
    concurrency: int = 5,
    state_builder: Callable[[Example], JSONContent] = default_state,
    client_factory: Callable[[str], tuple[Any, str]] | None = None,
) -> list[Path]:
    """Run the experiment against real providers. Refuses unless `approve_calls` equals the exact call count.

    `client_factory` exists so tests can inject fakes; by default it is `clients.factory.build_client`.
    Each provider's results go to a NEW parquet file; an existing file is never overwritten.
    """
    plan = plan_calls(experiment, rows, provider_specs)
    if approve_calls != plan.total_calls:
        raise StressError(f"Live run needs explicit approval of the exact call count: expected --approve-calls {plan.total_calls}, got {approve_calls}.")
    if client_factory is None:
        from clients.factory import build_client as client_factory  # imported lazily: keeps dry-run paths client-free

    examples = [row_to_example(r, dataset) for r in rows]
    if state_builder is default_state:  # rows may carry a structured `state_payload` (e.g. message + history)
        payloads = {r["variant_id"]: r["state_payload"] for r in rows if r.get("state_payload") is not None}
        state_builder = lambda e: payloads.get(e.example_id, default_state(e))  # noqa: E731
    written: list[Path] = []
    for spec in provider_specs:
        client, label = client_factory(spec)
        path = new_output_path(results_dir, label)  # checked before spending any calls
        evaluator = StressEvaluator(client, experiment=experiment, provider=label, state_builder=state_builder)
        async with client:
            results = await run_evaluation_async(evaluator, examples, concurrency=concurrency)
        config = ExperimentConfig(name=experiment, dataset=dataset, providers=(label,))
        results_dir.mkdir(parents=True, exist_ok=True)
        results_frame(results, rows, evaluator.probabilities, config).to_parquet(path, index=False)
        written.append(path)
    return written


# --- paired analysis -----------------------------------------------------------------------------


def pair_results(base: pd.DataFrame, variant: pd.DataFrame, on: Sequence[str] = ("provider", "source_example_id")) -> pd.DataFrame:
    """Join each variant row to its own base (clean/original) row. Columns get _base / _variant suffixes."""
    keep = ["prediction", "correct", "confidence", "latency_ms", "input_tokens", "error"]
    b = base[[*on, *[c for c in keep if c in base.columns]]].add_suffix("_base").rename(columns={f"{c}_base": c for c in on})
    v = variant.rename(columns={c: f"{c}_variant" for c in keep if c in variant.columns})
    dupes = b.duplicated(list(on)).sum()
    if dupes:
        raise StressError(f"{dupes} base row(s) share a pairing key {list(on)}; pairing would be ambiguous.")
    return v.merge(b, on=list(on), how="inner")


def paired_metrics(paired: pd.DataFrame) -> dict[str, Any]:
    """Metrics over pairs where BOTH sides produced a prediction. `flip` = the predicted label changed."""
    ok = paired[paired["prediction_base"].notna() & paired["prediction_variant"].notna()]
    n = len(ok)
    if n == 0:
        return {"n_pairs": 0, "n_excluded_errors": len(paired)}
    flips = ok["prediction_base"] != ok["prediction_variant"]
    scored = ok[ok["correct_base"].notna() & ok["correct_variant"].notna()]
    delta = ok["confidence_variant"] - ok["confidence_base"]
    return {
        "n_pairs": n,
        "n_excluded_errors": len(paired) - n,
        "label_agreement": float(1 - flips.mean()),
        "decision_flip_rate": float(flips.mean()),
        "accuracy_base": float(scored["correct_base"].astype(bool).mean()) if len(scored) else None,
        "accuracy_variant": float(scored["correct_variant"].astype(bool).mean()) if len(scored) else None,
        "accuracy_delta": float(scored["correct_variant"].astype(bool).mean() - scored["correct_base"].astype(bool).mean()) if len(scored) else None,
        "n_scored_pairs": len(scored),
        "mean_confidence_base": float(ok["confidence_base"].mean()),
        "mean_confidence_variant": float(ok["confidence_variant"].mean()),
        "mean_confidence_delta": float(delta.mean()),
        "mean_abs_confidence_delta": float(delta.abs().mean()),
    }


def grouped_paired_metrics(paired: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, grp in paired.groupby(list(by), dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **paired_metrics(grp)})
    return pd.DataFrame(rows)


def read_results(results_dir: Path) -> pd.DataFrame:
    """All raw_results_*.parquet files in a results dir, concatenated."""
    files = sorted(results_dir.glob("raw_results_*.parquet"))
    if not files:
        raise StressError(f"No raw_results_*.parquet in {results_dir}; no live results exist yet.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def write_tables(out_dir: Path, tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> list[Path]:
    """Write analysis tables (CSV) and summary.json. Refuses to overwrite."""
    paths = [out_dir / f"{name}.csv" for name in tables] + [out_dir / "summary.json"]
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise StressError(f"Refusing to overwrite existing analysis output(s): {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return paths


# --- generic CLI ---------------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    name: str
    dataset_dir: Path
    results_dir: Path
    build: Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]]
    describe: Callable[[list[dict[str, Any]]], dict[str, Any]]
    analyze: Callable[[pd.DataFrame], tuple[dict[str, pd.DataFrame], dict[str, Any]]]
    state_builder: Callable[[Example], JSONContent] = default_state
    dataset_name: str = "dataset"


def run_cli(spec: ExperimentSpec, argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=f"Phase II stress experiment: {spec.name}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Generate and freeze the dataset (refuses to overwrite).")
    report = sub.add_parser("report", help="Dry run: dataset counts and expected API calls. Makes no calls.")
    report.add_argument("--providers", default="jev")
    run = sub.add_parser("run", help="LIVE evaluation. Requires --approve-calls equal to the expected call count.")
    run.add_argument("--providers", default="jev")
    run.add_argument("--approve-calls", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=5)
    sub.add_parser("analyze", help="Offline analysis of recorded results.")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            rows, meta = spec.build()
            manifest = write_frozen_dataset(spec.dataset_dir, rows, meta, spec.dataset_name)
            print(json.dumps({"written": str(spec.dataset_dir), "n_rows": manifest["n_rows"], "sha256": manifest["sha256"], **spec.describe(rows)}, indent=2, default=str))
        elif args.command == "report":
            rows, manifest = load_frozen_dataset(spec.dataset_dir, spec.dataset_name)
            providers = [p.strip() for p in args.providers.split(",") if p.strip()]
            plan = plan_calls(spec.name, rows, providers)
            print(json.dumps({"dataset_sha256": manifest["sha256"], "expected_api_calls": plan.total_calls, "calls_per_provider": plan.calls_per_provider, "providers": providers, **spec.describe(rows)}, indent=2, default=str))
            print("DRY RUN: no client was constructed and no API call was made.")
        elif args.command == "run":
            rows, _ = load_frozen_dataset(spec.dataset_dir, spec.dataset_name)
            providers = [p.strip() for p in args.providers.split(",") if p.strip()]
            paths = asyncio.run(execute_live(spec.name, rows, f"stress:{spec.name}", providers, spec.results_dir, args.approve_calls, args.concurrency, spec.state_builder))
            print("wrote:", *paths, sep="\n  ")
        else:
            tables, summary = spec.analyze(read_results(spec.results_dir))
            print("wrote:", *write_tables(spec.results_dir, tables, summary), sep="\n  ")
    except StressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
