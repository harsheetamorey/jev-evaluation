# jev-decision-lab

Is Jev just a fast classifier, or is typed probabilistic judgment actually
useful as a programming primitive between deterministic code and generative
LLMs?

This project answers that with real labeled datasets (Bitext Customer
Support MCQ, MASSIVE), a shared `predict(example) -> PredictionResult`
interface across rules/Jev/LLM baselines, and a Streamlit dashboard for
comparing accuracy, latency, confidence, cost, and errors.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in TYPESAFE_API_KEY
uv run pytest
uv run python scripts/smoke_test.py   # one live Jev call, prints choice + latency
uv run python scripts/build_bitext_samples.py   # (re)generate data/samples/bitext_*.jsonl
uv run python src/experiments/bitext_hard_choice.py   # Experiment A: hard-choice classification via Jev
uv run python scripts/build_massive_samples.py   # (re)generate data/samples/massive_aligned_*.jsonl
uv run python src/experiments/multilingual.py   # Experiment B: multilingual classification via Jev

# Jev vs. a specific LLM (needs OPENAI_API_KEY and/or GEMINI_API_KEY in .env):
uv run python src/experiments/bitext_hard_choice.py --providers jev,openai:gpt-4o-mini
uv run python src/experiments/multilingual.py --providers jev,gemini:gemini-2.0-flash --locales en-US,hi-IN

uv run python scripts/build_choice_scaling_sample.py   # (re)generate data/samples/massive_choice_scaling*
uv run python src/experiments/choice_scaling.py   # Experiment C: accuracy vs. candidate-set size (K)

uv run python src/experiments/fanout.py   # Experiment D: latency vs. number of simultaneous questions (not accuracy)

uv run python src/experiments/confidence.py   # Experiment E: confidence vs. correctness (run bitext_hard_choice.py first)

uv run python src/experiments/routing_demo.py "I want to cancel my order"   # Experiment F: confidence-based routing (Jev fast path, LLM fallback)

uv run python src/experiments/contract_evolution.py   # Experiment G: format reliability vs. decision correctness

uv run streamlit run app/streamlit_app.py   # interactive dashboard + Try It Yourself playground
```

Cost estimates (`estimated_cost_usd`) come from recorded token counts times the
per-model rates in `src/evaluation/pricing.py` (each entry says where its rate
came from). They are ESTIMATES, never billed amounts, and a model with no entry
gets `null` rather than a guess.

## Dashboard

`app/streamlit_app.py` has 17 tabs. The 8 Phase I tabs come first (Overview,
Hard Choices, Multilingual, Choice Scaling, Parallel Decisions, Jev vs LLM,
Confidence, Try It Yourself); the 9 Phase II tabs follow (Calibration, Risk /
Coverage, Stability, Context Stress, OOD, Adversarial, Multilingual
Reliability, Failure Museum, Cascade Simulator). Try It Yourself is the only
tab that can make a live call (with your own `TYPESAFE_API_KEY`, on a button
press). The Phase II tabs only READ precomputed artifacts under
`data/results/phase2/` and never call a model; every tab that reads recorded
results says which command to run when nothing is recorded yet.

## What Part I tested

Experiments A-G above, on the Bitext Customer Support MCQ (1,000 main sample)
and MASSIVE (100 aligned utterances x 8 languages) datasets, with Jev as the
primary model and `gpt-4o-mini` as an optional reference: hard-choice
accuracy, multilingual accuracy, accuracy vs candidate-set size, latency vs
simultaneous questions, confidence vs correctness, confidence-based routing,
and contract evolution. Raw results are in `data/results/` (git-ignored).
Findings: `docs/part-i-findings.md`.

## What Part II tested

Part II asks what Part I could not: is Jev's confidence trustworthy, how does
it behave under stress, and how do architectures trade cost against quality?
Everything runs from a FROZEN baseline; nothing overwrites Part I results.

| Step | What | Command (`uv run python ...`) | Live API calls? | Output |
|---|---|---|---|---|
| 1 | Freeze the Part I baseline + a canonical (deduplicated) view | `src/phase2/freeze_baseline.py create` / `verify` | no | `data/results/phase2/baseline/` |
| 2 | Confidence calibration (ECE, correctness Brier, reliability bins) | `src/phase2/run_calibration.py` | no | `.../calibration/` |
| 3 | Selective prediction / risk-coverage | `src/phase2/run_selective.py` | no | `.../selective_prediction/` |
| 4-11 | Stress experiments: `run_ambiguity.py`, `run_stability.py`, `run_noise.py`, `run_context_pollution.py`, `run_context_relevance.py`, `run_ood.py`, `run_adversarial.py`, `run_choice_overlap.py` (each: `build`, `report`, `run`, `analyze`) | `build` / `report` / `analyze`: no. **`run`: YES** | `data/stress/<name>/` (frozen, tracked); results in `data/results/phase2/<name>/` |
| 12 | Multilingual reliability deep dive | `src/phase2/run_multilingual.py` | no | `.../multilingual/` |
| 13 | Failure taxonomy / Failure Museum data | `src/phase2/run_failures.py` | no | `.../failures/` |
| 14 | Jev -> LLM cascade simulation (offline) | `src/phase2/run_cascade.py` | no | `.../cascade/` |
| 15 | Rules -> Jev -> LLM -> oracle cascade (offline) | `src/phase2/run_full_cascade.py` | no | `.../full_cascade/` |
| 16 | Cost x quality comparison across architectures | `src/phase2/run_cost_quality.py` | no | `.../cost_quality/` |
| 17 | Dashboard tabs | `streamlit run app/streamlit_app.py` | no (except Try It Yourself) | - |
| 18 | Integrity / reproducibility | `src/phase2/verify_phase2.py verify` / `plan` / `manifest` | no | `.../phase2_manifest.json`, `docs/phase2-live-reproduction.md` |

Only the stress experiments' `run` command makes API calls, and it refuses
unless `--approve-calls` equals the exact expected call count (`report`
prints it and constructs no client). Status: the frozen stress datasets and
all analysis code exist; the live stress runs have NOT been executed, so the
stress tabs are empty until they are. Exact counts, estimated costs and
commands: `docs/phase2-live-reproduction.md`.

### Verify the frozen baseline and reproducibility

```bash
uv run python src/phase2/freeze_baseline.py verify   # frozen files, canonical view, example IDs
uv run python src/phase2/verify_phase2.py verify     # Phase I unchanged, byte-for-byte regeneration of every
                                                     # analysis artifact, stress datasets, offline pipelines
```

Run tests without spending API calls by blanking the keys:
`TYPESAFE_API_KEY= OPENAI_API_KEY= uv run pytest`. With a key present, the
key-gated tests (`test_jev_client_smoke`, `test_client_factory`) make a real
Jev request.

### Methodology caveats

- Part I did not record the resolved Jev model version, SDK version,
  concurrency, dataset revisions or sample seeds. The baseline manifest marks
  them `not_recorded`; current values are kept separately and never
  back-filled as history.
- Four Jev choice-scaling runs contained repeated calls. Raw files are kept
  byte-identical; the canonical view keeps the latest row per identity
  (independent of outcome) and the 400 repeats are documented in
  `canonical_dedup_report.csv` (8 disagreed, i.e. repeat agreement 98%).
- MASSIVE rows are not independent: the 8 locales are aligned translations of
  100 utterances, so n=800 per provider is closer to 100 effective examples.
- Calibration uses only the probability of the CHOSEN label (2 decimals); the
  Brier score is the binary correctness score, not multiclass. `gpt-4o-mini`
  confidence semantics were not investigated and it is a reference only.
- The "human" in the full cascade is an ORACLE returning the ground-truth
  label; it is not a measured human. The repository has no rules baseline
  (`src/clients/rules_client.py` is empty), so rule coverage is 0%.
- Cascade cost is an estimate from recorded tokens times `pricing.py`; the Jev
  rate is from the owner's notes and unverified. Cascade latency is a simulated
  sequential sum of measured per-call latencies, not a measured cascade.
- Paraphrases (`stability`) and context triplets (`context_relevance`) are
  hand-authored, frozen, and marked `pending_human_review`. The OOD source is
  MASSIVE en-US utterances (already local); AG News was not downloaded. No
  context-window limit is assumed for Jev; context budgets are configurable.
- Five frozen datasets do not store their selection seed in their manifest
  (the rows are frozen and rebuild deterministically; the seed constants are
  recorded in `phase2_manifest.json`).
- Thresholds, tags and cascade settings are configuration and slicing choices,
  not recommended operating points, and no "best" architecture is selected.
