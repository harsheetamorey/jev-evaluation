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
```

Cost estimates (`estimated_cost_usd`) are `null` until you fill in real,
current rates per model in `src/evaluation/pricing.py` -- nothing is guessed.





