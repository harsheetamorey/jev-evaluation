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
```





