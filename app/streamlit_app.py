"""Interactive dashboard + playground for jev-decision-lab.

Tabs: Overview, Hard Choices, Multilingual, Choice Scaling, Parallel
Decisions, Jev vs LLM, Confidence, Try It Yourself. Reads from the shared
data/results/results.parquet (written by the experiment scripts) and
data/results/fanout_results.parquet (fan-out has its own schema -- see
experiments/fanout.py). Every tab handles the "no data recorded yet" case
gracefully rather than crashing, since results only exist once you've run
an experiment with a real TYPESAFE_API_KEY.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from typesafe_sdk import Choice  # noqa: E402

from clients.jev_client import AsyncJevClient  # noqa: E402
from dataset_loaders.massive import INTENTS as MASSIVE_INTENTS  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_cross_language_consistency,
    compute_metrics,
    compute_metrics_by_group,
)
from evaluation.recorder import DEFAULT_RESULTS_PATH  # noqa: E402
from evaluation.runner import SystemOneEvaluator  # noqa: E402
from experiments.confidence import (  # noqa: E402
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    confidence_distribution_by_correctness,
    quadrant_examples,
)
from experiments.fanout import QUESTION_SPECS as FANOUT_QUESTION_SPECS  # noqa: E402
from experiments.fanout import aggregate_results as aggregate_fanout_results  # noqa: E402
from experiments.fanout import build_questions as build_fanout_questions  # noqa: E402
from experiments.routing_demo import DEFAULT_CANDIDATES as ROUTING_DEFAULT_CANDIDATES  # noqa: E402
from experiments.routing_demo import execute_route  # noqa: E402
from models.prediction import Example  # noqa: E402

FANOUT_RESULTS_PATH = DEFAULT_RESULTS_PATH.parent / "fanout_results.parquet"

PRIMARY_METRIC_COLUMNS = {
    "accuracy": "Accuracy",
    "p50_latency_ms": "p50 latency (ms)",
    "p95_latency_ms": "p95 latency (ms)",
    "error_rate": "Error rate",
    "total_estimated_cost_usd": "Cost (USD)",
}


# ---------------------------------------------------------------------------
# Data loading -- pure functions, testable without Streamlit.
# ---------------------------------------------------------------------------


def load_results_df(path: Path = DEFAULT_RESULTS_PATH) -> pd.DataFrame | None:
    """Load the shared results Parquet, or None if it doesn't exist yet."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def load_fanout_df(path: Path = FANOUT_RESULTS_PATH) -> pd.DataFrame | None:
    """Load fan-out's separate results Parquet (different schema), or None."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def filter_experiment(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    return df[df["experiment"] == experiment]


def filter_experiment_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[df["experiment"].str.startswith(prefix)]


def mean_choices_by_group(df: pd.DataFrame, group_col: str) -> dict[str, float]:
    """Mean number of candidate choices offered, per group -- "decision complexity"."""
    return df.groupby(group_col)["candidates"].apply(lambda s: s.apply(len).mean()).to_dict()


def metrics_table(by_group: dict[str, dict[str, Any]], choices_by_group: dict[str, float] | None = None) -> pd.DataFrame:
    """Turn compute_metrics_by_group()'s output into a display-ready DataFrame."""
    rows = []
    for key, metrics in by_group.items():
        row = {"": key}
        for col, label in PRIMARY_METRIC_COLUMNS.items():
            row[label] = metrics.get(col)
        row["Confidence (mean, correct)"] = metrics.get("confidence_when_correct")
        row["Confidence (mean, incorrect)"] = metrics.get("confidence_when_incorrect")
        if choices_by_group is not None:
            row["# choices"] = choices_by_group.get(key)
        row["n"] = metrics.get("n")
        rows.append(row)
    return pd.DataFrame(rows).set_index("")


def choice_scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the K-vs-accuracy tidy table straight from recorded results."""
    df = df.copy()
    df["num_choices"] = df["candidates"].apply(len)
    rows = []
    for k, group in df.groupby("num_choices"):
        m = compute_metrics(group)
        confidence = group["confidence"].dropna()
        rows.append(
            {
                "num_choices": k,
                "accuracy": m["accuracy"],
                "p50_latency_ms": m["p50_latency_ms"],
                "p95_latency_ms": m["p95_latency_ms"],
                "mean_confidence": float(confidence.mean()) if not confidence.empty else None,
                "n": m["n"],
            }
        )
    return pd.DataFrame(rows).sort_values("num_choices")


def overview_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """The primary cross-experiment metrics table for the Overview tab."""
    return metrics_table(compute_metrics_by_group(df, "experiment"), mean_choices_by_group(df, "experiment"))


# ---------------------------------------------------------------------------
# Rendering -- Streamlit calls live here; kept thin, delegating to the data
# functions above wherever there's real logic worth testing in isolation.
# ---------------------------------------------------------------------------


def _no_data_message(command: str) -> None:
    st.info(f"No results recorded yet. Run:\n\n```\nuv run python {command}\n```\n\nthen reload this page.")


def _download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False), file_name=filename, mime="text/csv")


def render_overview(df: pd.DataFrame | None) -> None:
    st.header("Overview")
    st.markdown(
        "Is Jev just a fast classifier, or is typed probabilistic judgment actually useful as a "
        "programming primitive between deterministic code and generative LLMs? This dashboard "
        "summarizes the recorded results from every experiment in `src/experiments/`."
    )
    st.markdown("**TEXT → JEV → TYPED DECISION → CODE** -- see the *Try It Yourself* tab to run this live.")

    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return

    st.subheader("Primary metrics, by experiment")
    st.caption("Accuracy: was the decision right? · p50/p95: latency · Error rate: API/schema/reliability · Cost: practical tradeoff.")
    st.dataframe(overview_dashboard(df))
    st.metric("Total recorded predictions", len(df))
    _download_button(df, "Download all raw results (CSV)", "results.csv")


def render_hard_choices(df: pd.DataFrame | None) -> None:
    st.header("Hard Choices (Bitext)")
    st.caption("5-way choice among semantically close distractors. # choices = 5 for every row here.")
    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return
    subset = filter_experiment(df, "bitext_hard_choice")
    if subset.empty:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return
    st.dataframe(metrics_table(compute_metrics_by_group(subset, "provider"), mean_choices_by_group(subset, "provider")))
    _download_button(subset, "Download raw results (CSV)", "bitext_hard_choice.csv")


def render_multilingual(df: pd.DataFrame | None) -> None:
    st.header("Multilingual (MASSIVE, aligned)")
    st.caption("Same underlying utterance, evaluated across languages -- accuracy by Language.")
    if df is None:
        _no_data_message("src/experiments/multilingual.py")
        return
    subset = filter_experiment(df, "multilingual")
    if subset.empty:
        _no_data_message("src/experiments/multilingual.py")
        return
    st.subheader("By provider")
    st.dataframe(metrics_table(compute_metrics_by_group(subset, "provider")))
    st.subheader("By language (locale)")
    st.dataframe(metrics_table(compute_metrics_by_group(subset, "locale")))
    st.subheader("Cross-language consistency")
    st.caption("For each aligned utterance: how many locales' predictions agreed with ground truth.")
    consistency = compute_cross_language_consistency(subset)
    st.bar_chart(pd.Series(consistency["histogram"], name="count of aligned utterances"))
    _download_button(subset, "Download raw results (CSV)", "multilingual.csv")


def render_choice_scaling(df: pd.DataFrame | None) -> None:
    st.header("Choice Scaling")
    st.caption("What happens to accuracy and latency as the candidate-set size (K) grows? Not correctness across K, that's this tab's point.")
    if df is None:
        _no_data_message("src/experiments/choice_scaling.py")
        return
    subset = filter_experiment_prefix(df, "choice_scaling")
    if subset.empty:
        _no_data_message("src/experiments/choice_scaling.py")
        return
    table = choice_scaling_table(subset)
    st.dataframe(table.set_index("num_choices"))
    st.line_chart(table.set_index("num_choices")[["accuracy"]])
    st.line_chart(table.set_index("num_choices")[["p50_latency_ms", "p95_latency_ms"]])
    _download_button(subset, "Download raw results (CSV)", "choice_scaling.csv")


def render_parallel_decisions(fanout_df: pd.DataFrame | None) -> None:
    st.header("Parallel Decisions (Fan-out)")
    st.caption("One message, many simultaneous questions in one call. Latency behavior only -- most questions have no ground truth, so no accuracy is shown here.")
    if fanout_df is None:
        _no_data_message("src/experiments/fanout.py")
        return
    summary = aggregate_fanout_results(fanout_df)
    st.dataframe(summary.set_index("num_questions"))
    st.line_chart(summary.set_index("num_questions")[["total_latency_ms", "latency_per_question_ms"]])
    _download_button(fanout_df, "Download raw results (CSV)", "fanout_results.csv")


def render_jev_vs_llm(df: pd.DataFrame | None) -> None:
    st.header("Jev vs LLM")
    st.caption("Every column is headed by an explicit model name -- never a generic \"LLM\".")
    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py --providers jev,openai:gpt-4o-mini")
        return
    experiments = sorted(df["experiment"].unique())
    if not experiments:
        _no_data_message("src/experiments/bitext_hard_choice.py --providers jev,openai:gpt-4o-mini")
        return
    chosen = st.selectbox("Experiment", experiments)
    subset = filter_experiment(df, chosen)
    providers = sorted(subset["provider"].unique())
    if len(providers) < 2:
        st.info(f"Only one provider ({providers[0] if providers else 'none'}) has recorded results for {chosen!r}. "
                f"Run it again with e.g. --providers jev,openai:gpt-4o-mini to compare.")
    st.dataframe(metrics_table(compute_metrics_by_group(subset, "provider"), mean_choices_by_group(subset, "provider")))
    _download_button(subset, "Download raw results (CSV)", f"{chosen}_by_provider.csv")


def render_confidence(df: pd.DataFrame | None) -> None:
    st.header("Confidence")
    st.caption("Exploratory only (Part I) -- not calibration. Can we trust this confidence? Part II's question.")
    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return
    scored = df[df["correct"].notna() & df["confidence"].notna()]
    if scored.empty:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return

    st.subheader("Confidence distribution: correct vs. incorrect")
    st.dataframe(confidence_distribution_by_correctness(scored).set_index("correct"))

    bins = pd.cut(scored["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(str)
    histogram = scored.groupby([bins, "correct"], observed=True).size().unstack(fill_value=0)
    st.bar_chart(histogram)

    st.subheader("Example predictions by quadrant")
    quadrants = quadrant_examples(scored, high=HIGH_CONFIDENCE_THRESHOLD, low=LOW_CONFIDENCE_THRESHOLD)
    for name, rows in quadrants.items():
        with st.expander(f"{name.replace('_', ' ').title()} ({len(rows)})"):
            st.dataframe(rows)
    _download_button(scored, "Download scored results (CSV)", "confidence.csv")


async def _classify_choice(message: str, candidates: list[str]) -> dict:
    example = Example(example_id="try-it", dataset="try-it", state=message, candidates=candidates)
    async with AsyncJevClient() as client:
        evaluator = SystemOneEvaluator(client, experiment="try_it_yourself", provider="jev")
        return await evaluator.predict(example)


async def _classify_parallel(message: str, n_questions: int) -> tuple[dict, float]:
    questions = build_fanout_questions(n_questions)
    start = time.perf_counter()
    async with AsyncJevClient() as client:
        response = await client.system_one(state={"user_message": message}, questions=questions)
    latency_ms = (time.perf_counter() - start) * 1000
    return response, latency_ms


def render_try_it_yourself() -> None:
    st.header("Try It Yourself")
    st.markdown("**TEXT → JEV → TYPED DECISION → CODE**")

    text = st.text_area("Message", placeholder="e.g. I want to cancel my order and get a refund")
    mode = st.radio(
        "Mode",
        ["Customer-support choices", "MASSIVE 60-intent routing", "Parallel judgments"],
    )

    if mode == "Parallel judgments":
        n_questions = st.slider("Number of simultaneous questions", 1, len(FANOUT_QUESTION_SPECS), 10)

    if st.button("Classify", disabled=not text.strip()):
        col_text, col_jev, col_decision, col_code = st.columns(4)
        col_text.info(f"**TEXT**\n\n{text}")
        col_jev.info("**JEV**\n\ncalling...")

        try:
            if mode == "Customer-support choices":
                result = asyncio.run(_classify_choice(text, ROUTING_DEFAULT_CANDIDATES))
                col_jev.success("**JEV**\n\ndone")
                col_decision.info(f"**TYPED DECISION**\n\n{result.prediction}\n\nconfidence: {result.confidence}")
                st.metric("Prediction", result.prediction)
                st.metric("Confidence", result.confidence)
                st.metric("Latency (ms)", round(result.latency_ms, 1))
                st.bar_chart(pd.Series({c: (1.0 if c == result.prediction else 0.0) for c in ROUTING_DEFAULT_CANDIDATES}, name="probabilities (from confidence, top choice only)"))
                action = execute_route(result.prediction) if result.prediction else "no action (error)"
                col_code.info(f"**CODE**\n\nSoftware action:\n\n{action}")
            elif mode == "MASSIVE 60-intent routing":
                result = asyncio.run(_classify_choice(text, MASSIVE_INTENTS))
                col_jev.success("**JEV**\n\ndone")
                col_decision.info(f"**TYPED DECISION**\n\n{result.prediction}\n\nconfidence: {result.confidence}")
                st.metric("Prediction", result.prediction)
                st.metric("Confidence", result.confidence)
                st.metric("Latency (ms)", round(result.latency_ms, 1))
                action = execute_route(result.prediction) if result.prediction else "no action (error)"
                col_code.info(f"**CODE**\n\nSoftware action:\n\n{action}")
            else:
                response, latency_ms = asyncio.run(_classify_parallel(text, n_questions))
                col_jev.success("**JEV**\n\ndone")
                answers = []
                for name, answer in response.answers.items():
                    value = getattr(answer, "choice", None) or getattr(answer, "score", None) or getattr(answer, "noul", None)
                    answers.append({"question": name, "answer": value})
                answers_df = pd.DataFrame(answers)
                col_decision.info(f"**TYPED DECISION**\n\n{len(answers)} answers")
                st.dataframe(answers_df)
                st.metric("Total latency (ms)", round(latency_ms, 1))
                st.metric("Latency per question (ms)", round(latency_ms / n_questions, 1))
                col_code.info(f"**CODE**\n\n{len(answers)} typed answers available to branch on")
        except Exception as exc:  # noqa: BLE001 - surface any failure (e.g. missing API key) directly in the UI
            col_jev.error(f"**JEV**\n\n{exc}")


def main() -> None:
    st.set_page_config(page_title="jev-decision-lab", layout="wide")
    st.title("jev-decision-lab")

    results_df = load_results_df()
    fanout_df = load_fanout_df()

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    tabs = st.tabs(
        [
            "Overview",
            "Hard Choices",
            "Multilingual",
            "Choice Scaling",
            "Parallel Decisions",
            "Jev vs LLM",
            "Confidence",
            "Try It Yourself",
        ]
    )
    with tabs[0]:
        render_overview(results_df)
    with tabs[1]:
        render_hard_choices(results_df)
    with tabs[2]:
        render_multilingual(results_df)
    with tabs[3]:
        render_choice_scaling(results_df)
    with tabs[4]:
        render_parallel_decisions(fanout_df)
    with tabs[5]:
        render_jev_vs_llm(results_df)
    with tabs[6]:
        render_confidence(results_df)
    with tabs[7]:
        render_try_it_yourself()


if __name__ == "__main__":
    main()
