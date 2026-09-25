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

import altair as alt  # noqa: E402
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
    accuracy_at_thresholds,
    confidence_distribution_by_correctness,
    confidence_histogram,
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

CYAN = "#22d3ee"
TEAL = "#2dd4bf"
SKY = "#38bdf8"
AMBER = "#fbbf24"
ROSE = "#fb7185"
SLATE = "#94a3b8"
TEXT = "#e8eef7"

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: "IBM Plex Sans", sans-serif; }

.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }

h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; }
h2 { font-weight: 700 !important; border-bottom: 1px solid rgba(34, 211, 238, 0.18);
     padding-bottom: 0.45rem; margin-top: 0.4rem !important; }
h3 { font-weight: 700 !important; }

div[data-testid="stTabs"] button { font-weight: 600; }

.hero-kicker {
    color: #67e8f9; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; margin-bottom: 0.2rem;
}
.hero-sub { color: #94a3b8; margin-top: -0.4rem; margin-bottom: 1.1rem; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; margin: 0.4rem 0 1rem; }
.kpi {
    background: #101e30;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px; padding: 0.8rem 0.9rem 0.7rem;
}
.kpi .label { color: #8ea3bd; font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.kpi .value { color: #e8eef7; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.2; margin: 0.22rem 0 0.1rem; }
.kpi .hint { color: #7f96b0; font-size: 0.76rem; line-height: 1.35; }

.meaning {
    background: rgba(15, 30, 50, 0.7);
    border-left: 2px solid rgba(34, 211, 238, 0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.85rem;
    color: #b9cadd;
    font-size: 0.89rem;
    line-height: 1.5;
    margin: 0.15rem 0 0.75rem;
}
.meaning b { color: #dbe6f3; }
.callout {
    background: rgba(16, 34, 52, 0.8);
    border: 1px solid rgba(34, 211, 238, 0.22);
    border-radius: 10px; padding: 0.7rem 0.95rem; color: #cfe3f0;
    font-size: 0.9rem; line-height: 1.5; margin: 0.4rem 0 0.85rem;
}
.takeaway {
    background: #0f2233; border: 1px solid rgba(45, 212, 191, 0.28);
    border-radius: 12px; padding: 0.85rem 1.05rem; color: #bfe9e1;
    font-size: 0.93rem; line-height: 1.5; margin-top: 0.9rem;
}
.takeaway b { color: #e6fffa; }
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 0.6rem 0 1rem; }
.pipe {
    background: #101e30; border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px; padding: 0.65rem 0.8rem;
}
.pipe .step { color: #67e8f9; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.13em; }
.pipe .body { color: #dbe6f3; font-size: 0.87rem; margin-top: 0.22rem; line-height: 1.35; }

/* --- Section header: numbered badge + title + one-line framing --------- */
.sec { display: flex; align-items: flex-start; gap: 0.65rem; margin: 1.4rem 0 0.5rem; }
.sec .num {
    flex: 0 0 auto; width: 25px; height: 25px; border-radius: 7px; margin-top: 2px;
    background: #16293f; border: 1px solid rgba(125, 211, 252, 0.22); color: #7dd3fc;
    font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.sec .t { color: #e8eef7; font-size: 1.0rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.3; }
.sec .s { color: #8ea3bd; font-size: 0.84rem; margin-top: 0.12rem; line-height: 1.4; }

/* --- "How to read this chart": the axes spelled out, never implied ----- */
.howto { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 0.4rem; margin: 0.3rem 0 0.6rem; }
.howto .cell {
    background: rgba(16, 30, 48, 0.6); border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 8px; padding: 0.45rem 0.6rem;
}
.howto .k { color: #7f96b0; font-size: 0.63rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.howto .v { color: #cfdbea; font-size: 0.82rem; margin-top: 0.16rem; line-height: 1.4; }
.howto .v code { font-size: 0.78rem; }

/* --- Insight strip: the one sentence the chart above actually supports - */
.insight {
    display: flex; gap: 0.55rem; align-items: flex-start;
    background: rgba(18, 34, 54, 0.7); border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 9px; padding: 0.55rem 0.75rem; margin: 0.5rem 0 0.2rem;
    color: #c3d2e4; font-size: 0.87rem; line-height: 1.45;
}
.insight .ico { flex: 0 0 auto; opacity: 0.8; font-size: 0.85rem; line-height: 1.6; }
.insight b { color: #e8eef7; }
.insight.good { background: rgba(14, 40, 44, 0.6); border-color: rgba(45, 212, 191, 0.26); color: #b6e3da; }
.insight.good b { color: #e6fffa; }
.insight.warn { background: rgba(44, 36, 16, 0.55); border-color: rgba(251, 191, 36, 0.28); color: #e7d5a4; }
.insight.warn b { color: #fdf0cf; }

/* --- Inline stat chips: the numbers behind a panel, without a table ---- */
.chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.25rem 0 0.7rem; }
.chip {
    background: rgba(16, 30, 48, 0.75); border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 999px; padding: 0.24rem 0.68rem; font-size: 0.78rem; color: #93a8c0;
}
.chip b { color: #e8eef7; font-weight: 700; }

/* --- "This is the original view, kept" markers ------------------------- */
.kept {
    color: #6d829b; font-size: 0.78rem; font-style: italic;
    border-top: 1px dashed rgba(148, 163, 184, 0.18); padding-top: 0.4rem; margin: 0.15rem 0 0.9rem;
}

@media (max-width: 1100px) {
    .kpi-grid, .pipeline { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""

def _metric_column_config() -> dict:
    """Built at call time so importing this module in pytest doesn't need a Streamlit runtime."""
    return {
        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.1%", help="Share of scored (non-error) rows whose prediction matched ground truth."),
        "Accuracy 95% CI": st.column_config.TextColumn("Accuracy 95% CI", help="Wilson 95% interval. Small n (e.g. 100/locale) means wide, overlapping intervals — do not rank neighbours from the point estimate alone."),
        "p50 latency (ms)": st.column_config.NumberColumn("p50 latency (ms)", format="%.0f", help="Median end-to-end API latency for this group."),
        "p95 latency (ms)": st.column_config.NumberColumn("p95 latency (ms)", format="%.0f", help="95th-percentile latency — the slow tail, not the typical call."),
        "Error rate": st.column_config.NumberColumn("Error rate", format="%.2%", help="Share of rows with an API/schema/transport error. Accuracy ignores these rows."),
        "Cost (USD)": st.column_config.NumberColumn("Cost (USD)", format="$%.4f", help="Sum of estimated_cost_usd from token counts × src/evaluation/pricing.py. Null if the model has no rate."),
        "Confidence (mean, correct)": st.column_config.NumberColumn("Confidence (mean, correct)", format="%.3f", help="Mean reported confidence among rows that were correct."),
        "Confidence (mean, incorrect)": st.column_config.NumberColumn("Confidence (mean, incorrect)", format="%.3f", help="Mean reported confidence among rows that were wrong. Separation vs. the correct column is the Part I confidence signal."),
        "# choices": st.column_config.NumberColumn("# choices", format="%.0f", help="Mean size of the candidate set offered to the model in this group."),
        "n": st.column_config.NumberColumn("n", format="%d", help="Row count for this group, including errors."),
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


def _format_ci(low: float | None, high: float | None) -> str | None:
    if low is None or high is None:
        return None
    return f"[{low:.1%}, {high:.1%}]"


def metrics_table(by_group: dict[str, dict[str, Any]], choices_by_group: dict[str, float] | None = None) -> pd.DataFrame:
    """Turn compute_metrics_by_group()'s output into a display-ready DataFrame.

    Accuracy's 95% Wilson CI is shown alongside it -- small samples (e.g. n=100
    per language) can have wide, overlapping intervals, so the point estimate
    alone can overstate how confidently groups can be ranked against each other.
    """
    rows = []
    for key, metrics in by_group.items():
        row = {"": key}
        for col, label in PRIMARY_METRIC_COLUMNS.items():
            row[label] = metrics.get(col)
        row["Accuracy 95% CI"] = _format_ci(metrics.get("accuracy_ci_low"), metrics.get("accuracy_ci_high"))
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
                "accuracy_95_ci": _format_ci(m["accuracy_ci_low"], m["accuracy_ci_high"]),
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


def choice_scaling_by_provider(df: pd.DataFrame) -> pd.DataFrame:
    """Same tidy K-table as choice_scaling_table, split by provider so mixed runs aren't silently pooled."""
    d = df.copy()
    d["num_choices"] = d["candidates"].apply(len)
    rows = []
    for (k, provider), group in d.groupby(["num_choices", "provider"]):
        m = compute_metrics(group)
        confidence = group["confidence"].dropna()
        rows.append(
            {
                "num_choices": int(k),
                "provider": str(provider),
                "accuracy": m["accuracy"],
                "accuracy_95_ci": _format_ci(m["accuracy_ci_low"], m["accuracy_ci_high"]),
                "p50_latency_ms": m["p50_latency_ms"],
                "p95_latency_ms": m["p95_latency_ms"],
                "mean_confidence": float(confidence.mean()) if not confidence.empty else None,
                "n": m["n"],
            }
        )
    return pd.DataFrame(rows).sort_values(["provider", "num_choices"])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.1%}"


def _fmt_num(value: float | None, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{digits}f}"


def _altair_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#1e3a5f",
            domainColor="#334155",
            labelColor=SLATE,
            titleColor=TEXT,
            tickColor="#334155",
            labelFont="IBM Plex Sans",
            titleFont="IBM Plex Sans",
        )
        .configure_legend(labelColor=TEXT, titleColor=TEXT, orient="bottom")
        .configure_title(color=TEXT, font="IBM Plex Sans", fontSize=14, anchor="start")
    )


def _kpi_html(cards: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>'
        for label, value, hint in cards
    )
    return f'<div class="kpi-grid">{cells}</div>'


def _meaning(text: str) -> None:
    st.markdown(f'<div class="meaning">{text}</div>', unsafe_allow_html=True)


def _callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def _takeaway(text: str) -> None:
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)


def _section(number: str, title: str, subtitle: str) -> None:
    """Numbered section header: a badge, the title, and one line saying why it exists."""
    st.markdown(
        f'<div class="sec"><div class="num">{number}</div>'
        f'<div><div class="t">{title}</div><div class="s">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def _how_to_read(items: list[tuple[str, str]]) -> None:
    """Spell the chart's axes out. Each item is (LABEL, explanation) -- e.g.
    ("X AXIS", "K = candidate-set size"), ("ONE POINT", "one K, one provider").
    """
    cells = "".join(f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items)
    st.markdown(f'<div class="howto">{cells}</div>', unsafe_allow_html=True)


def _insight(text: str, tone: str = "", icon: str = "→") -> None:
    """The single sentence the chart directly above actually supports.

    tone: "" (neutral), "good" (a claim the data backs), "warn" (a caveat or a
    failure mode). Kept separate from _meaning, which explains the measurement;
    this states the finding.
    """
    st.markdown(f'<div class="insight {tone}"><div class="ico">{icon}</div><div>{text}</div></div>', unsafe_allow_html=True)


def _chips(items: list[tuple[str, str]]) -> None:
    """Compact label/value pills for the handful of numbers behind a panel."""
    body = "".join(f"<div class=\"chip\">{label} <b>{value}</b></div>" for label, value in items)
    st.markdown(f'<div class="chips">{body}</div>', unsafe_allow_html=True)


def _kept(text: str) -> None:
    """Marks a duplicated original view that is deliberately retained, not dead weight."""
    st.markdown(f'<div class="kept">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Insight text derived from the loaded data -- so the sentence under a chart
# describes the Parquet actually on disk, not a number copied from a write-up
# that may no longer match after a re-run.
# ---------------------------------------------------------------------------


def locale_spread_insight(df: pd.DataFrame) -> str | None:
    """Best vs. worst locale, with the honest caveat about overlapping intervals."""
    jev = _prefer_jev(df)
    rows = []
    for locale, group in jev.groupby("locale"):
        m = compute_metrics(group)
        if m["accuracy"] is not None:
            rows.append((str(locale), m["accuracy"], m["n"], m["accuracy_ci_low"], m["accuracy_ci_high"]))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[1], reverse=True)
    (top, top_acc, top_n, top_lo, _), (bottom, bottom_acc, _, _, bottom_hi) = rows[0], rows[-1]
    gap = (top_acc - bottom_acc) * 100
    # Two intervals are separated when the worse locale's UPPER bound sits below
    # the better one's LOWER bound. Comparing the outer bounds instead (top's
    # high vs. bottom's low) can never be true and would silently hardcode the
    # "overlapping" branch.
    separated = top_lo is not None and bottom_hi is not None and bottom_hi < top_lo
    tail = (
        f"Those two ends do not overlap at 95%, so a real spread is supported — but adjacent bars still are not separable at n≈{top_n}."
        if separated
        else f"Even the two extremes have overlapping 95% intervals at n≈{top_n}, so read this as a spread, not a ranking."
    )
    return f"<b>{top}</b> leads at {top_acc:.0%} and <b>{bottom}</b> trails at {bottom_acc:.0%} — a {gap:.0f}-point spread. {tail}"


def choice_scaling_insight(tidy: pd.DataFrame) -> str | None:
    """Does accuracy fall as K grows, and does confidence fall with it or stay pinned?"""
    jev = tidy[tidy["provider"] == "jev"] if "jev" in set(tidy["provider"]) else tidy
    jev = jev.sort_values("num_choices")
    if len(jev) < 2:
        return None
    first, last = jev.iloc[0], jev.iloc[-1]
    # Signed as (last - first), so a decline prints as a negative delta. Computing
    # it the other way round labels a 16-point drop "+16", which reads as a gain.
    acc_delta = (last["accuracy"] - first["accuracy"]) * 100
    conf_first, conf_last = first["mean_confidence"], last["mean_confidence"]
    if conf_first is None or conf_last is None or pd.isna(conf_first) or pd.isna(conf_last):
        return f"Accuracy moves {first['accuracy']:.0%} → {last['accuracy']:.0%} from K={int(first['num_choices'])} to K={int(last['num_choices'])} ({acc_delta:+.0f} pts)."
    conf_delta = (conf_last - conf_first) * 100
    tracking = (
        "Confidence falls with it, so the model is at least registering the added difficulty."
        if conf_delta < -1
        else "Confidence barely moves, so the model is getting worse <b>without noticing</b> — the dangerous shape."
    )
    return (
        f"K={int(first['num_choices'])} → K={int(last['num_choices'])}: accuracy {first['accuracy']:.0%} → {last['accuracy']:.0%} "
        f"({acc_delta:+.0f} pts), mean confidence {conf_first:.2f} → {conf_last:.2f} ({conf_delta:+.0f} pts). {tracking}"
    )


def fanout_insight(summary: pd.DataFrame) -> str | None:
    """The fan-out claim in one line: total is ~flat, so per-question collapses."""
    jev = summary[summary["provider"] == "jev"] if "provider" in summary.columns and "jev" in set(summary["provider"]) else summary
    jev = jev.sort_values("num_questions")
    if len(jev) < 2:
        return None
    first, last = jev.iloc[0], jev.iloc[-1]
    total_ratio = last["total_latency_ms"] / first["total_latency_ms"] if first["total_latency_ms"] else None
    per_q_ratio = first["latency_per_question_ms"] / last["latency_per_question_ms"] if last["latency_per_question_ms"] else None
    n_first, n_last = int(first["num_questions"]), int(last["num_questions"])
    if total_ratio is None or per_q_ratio is None:
        return None
    return (
        f"{n_first} → {n_last} questions is {n_last // max(n_first, 1)}× the work, but total wall-clock only grows "
        f"{first['total_latency_ms']:.0f} → {last['total_latency_ms']:.0f} ms ({total_ratio:.1f}×). "
        f"Per-question cost therefore collapses {first['latency_per_question_ms']:.0f} → {last['latency_per_question_ms']:.0f} ms "
        f"({per_q_ratio:.0f}× cheaper) — that second number is arithmetic on the first, not a separate result."
    )


def confidence_gate_insight(gates: pd.DataFrame) -> str | None:
    """What the trust gate actually buys you, at the tightest threshold that still
    meets the routing demo's cutoff.

    Picked as the first threshold >= HIGH_CONFIDENCE_THRESHOLD rather than the
    nearest one: 0.85 is equidistant from the 0.80 and 0.90 rows in the default
    sweep, and quoting the looser of the two would overstate coverage for a gate
    that is supposed to be at least as strict as the demo's.
    """
    if gates.empty:
        return None
    at_or_above = gates[gates["threshold"] >= HIGH_CONFIDENCE_THRESHOLD]
    row = at_or_above.iloc[0] if not at_or_above.empty else gates.iloc[-1]
    if row["accuracy"] is None or pd.isna(row["accuracy"]):
        return None
    n_wrong = int(row["n_wrong"])
    return (
        f"Gate at confidence ≥ {row['threshold']:.2f} and you keep <b>{row['coverage']:.0%}</b> of traffic at "
        f"<b>{row['accuracy']:.1%}</b> accuracy — but {n_wrong:,} wrong predictions still clear the gate and would execute. "
        f"The remaining {1 - row['coverage']:.0%} is what you route to an LLM or a human."
    )


def _no_data_message(command: str) -> None:
    st.info(f"No results recorded yet. Run:\n\n```\nuv run python {command}\n```\n\nthen reload this page.")


def _download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False), file_name=filename, mime="text/csv")


def _show_table(table: pd.DataFrame, *, reset: bool = False) -> None:
    display = table.reset_index() if reset else table
    st.dataframe(display, use_container_width=True, hide_index=not reset, column_config=_metric_column_config())


def _prefer_jev(df: pd.DataFrame) -> pd.DataFrame:
    if "provider" in df.columns and (df["provider"] == "jev").any():
        return df[df["provider"] == "jev"]
    return df


def _locale_accuracy_chart(df: pd.DataFrame) -> alt.Chart:
    rows = []
    for (locale, provider), group in df.groupby(["locale", "provider"]):
        m = compute_metrics(group)
        rows.append({"locale": str(locale), "provider": str(provider), "accuracy": m["accuracy"], "n": m["n"], "ci": _format_ci(m["accuracy_ci_low"], m["accuracy_ci_high"])})
    chart_df = pd.DataFrame(rows)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("locale:N", sort="-x", title=None),
            color=alt.Color("provider:N", scale=alt.Scale(range=[CYAN, AMBER]), legend=alt.Legend(title="Provider")),
            yOffset="provider:N",
            tooltip=["locale", "provider", alt.Tooltip("accuracy:Q", format=".1%"), "n", "ci"],
        )
        .properties(height=max(220, 28 * chart_df["locale"].nunique()))
    )
    return _altair_theme(chart)


def _consistency_chart(histogram: dict[str, int]) -> alt.Chart:
    chart_df = pd.DataFrame({"locales_correct": list(histogram.keys()), "utterances": list(histogram.values())})
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=CYAN, cornerRadiusEnd=4)
        .encode(
            x=alt.X("locales_correct:N", title="Locales correct out of aligned locales", sort="-y"),
            y=alt.Y("utterances:Q", title="Aligned utterances"),
            tooltip=["locales_correct", "utterances"],
        )
        .properties(height=280)
    )
    return _altair_theme(chart)


def _choice_scaling_charts(tidy: pd.DataFrame) -> tuple[alt.Chart, alt.Chart]:
    color = alt.Color("provider:N", scale=alt.Scale(range=[CYAN, AMBER]), legend=alt.Legend(title="Provider"))
    acc = _altair_theme(
        alt.Chart(tidy)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X("num_choices:O", title="K = number of candidate intents"),
            y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0.5, 1.0]), axis=alt.Axis(format="%")),
            color=color,
            tooltip=["provider", "num_choices", alt.Tooltip("accuracy:Q", format=".1%"), "accuracy_95_ci", "n"],
        )
        .properties(height=260, title="Accuracy vs K")
    )
    conf = _altair_theme(
        alt.Chart(tidy)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X("num_choices:O", title="K = number of candidate intents"),
            y=alt.Y("mean_confidence:Q", title="Mean confidence", scale=alt.Scale(domain=[0.5, 1.0])),
            color=color,
            tooltip=["provider", "num_choices", alt.Tooltip("mean_confidence:Q", format=".3f"), "n"],
        )
        .properties(height=260, title="Mean confidence vs K")
    )
    return acc, conf


def _latency_chart(tidy: pd.DataFrame) -> alt.Chart:
    long = tidy.melt(
        id_vars=[c for c in ("num_choices", "provider") if c in tidy.columns],
        value_vars=["p50_latency_ms", "p95_latency_ms"],
        var_name="percentile",
        value_name="latency_ms",
    )
    encodings = {
        "x": alt.X("num_choices:O", title="K = number of candidate intents"),
        "y": alt.Y("latency_ms:Q", title="Latency (ms)"),
        "color": alt.Color("percentile:N", scale=alt.Scale(range=[SKY, ROSE]), legend=alt.Legend(title="Percentile")),
        "tooltip": ["num_choices", "percentile", alt.Tooltip("latency_ms:Q", format=".0f")],
    }
    if "provider" in long.columns:
        encodings["strokeDash"] = alt.StrokeDash("provider:N", legend=alt.Legend(title="Provider"))
        encodings["tooltip"] = ["provider", "num_choices", "percentile", alt.Tooltip("latency_ms:Q", format=".0f")]
    return _altair_theme(alt.Chart(long).mark_line(point=True).encode(**encodings).properties(height=260, title="Latency vs K (should stay nearly flat)"))


def _fanout_charts(summary: pd.DataFrame) -> tuple[alt.Chart, alt.Chart]:
    color_enc = alt.Color("provider:N", scale=alt.Scale(range=[CYAN, AMBER]), legend=alt.Legend(title="Provider")) if "provider" in summary.columns else alt.value(CYAN)
    tooltip = [c for c in ("provider", "num_questions", "total_latency_ms", "latency_per_question_ms", "input_tokens", "output_tokens") if c in summary.columns]
    total = _altair_theme(
        alt.Chart(summary)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_questions:O", title="Simultaneous questions in one call"),
            y=alt.Y("total_latency_ms:Q", title="Total wall-clock (ms)"),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(height=280, title="Total latency — the waiting the user feels")
    )
    per_q = _altair_theme(
        alt.Chart(summary)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_questions:O", title="Simultaneous questions in one call"),
            y=alt.Y("latency_per_question_ms:Q", title="ms / question  (= total ÷ n)"),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(height=280, title="Per-question latency — collapses if total stays flat")
    )
    return total, per_q


def _token_chart(summary: pd.DataFrame) -> alt.Chart:
    id_vars = [c for c in ("num_questions", "provider") if c in summary.columns]
    long = summary.melt(id_vars=id_vars, value_vars=[c for c in ("input_tokens", "output_tokens") if c in summary.columns], var_name="token_type", value_name="tokens")
    return _altair_theme(
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_questions:O", title="Simultaneous questions"),
            y=alt.Y("tokens:Q", title="Mean tokens per call"),
            color=alt.Color("token_type:N", scale=alt.Scale(range=[SKY, TEAL]), legend=alt.Legend(title="Tokens")),
            strokeDash=alt.StrokeDash("provider:N") if "provider" in long.columns else alt.value([1]),
            tooltip=id_vars + ["token_type", alt.Tooltip("tokens:Q", format=".0f")],
        )
        .properties(height=260, title="Tokens still grow — this is why we say latency, not cost")
    )


def _confidence_hist_chart(histogram: pd.DataFrame) -> alt.Chart:
    long = histogram.reset_index() if histogram.index.name else histogram.copy()
    if "confidence_bin" not in long.columns:
        long = long.reset_index()
    value_cols = [c for c in long.columns if c in ("Correct", "Incorrect", True, False)]
    rename = {True: "Correct", False: "Incorrect"}
    long = long.rename(columns=rename)
    value_cols = [c for c in ("Correct", "Incorrect") if c in long.columns]
    melted = long.melt(id_vars=["confidence_bin"], value_vars=value_cols, var_name="outcome", value_name="count")
    return _altair_theme(
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("confidence_bin:N", title="Confidence bin (right-inclusive except the first)"),
            y=alt.Y("count:Q", title="Predictions"),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["Correct", "Incorrect"], range=[CYAN, ROSE]), legend=alt.Legend(title=None)),
            xOffset="outcome:N",
            tooltip=["confidence_bin", "outcome", "count"],
        )
        .properties(height=280)
    )


def _gate_chart(gates: pd.DataFrame) -> alt.Chart:
    long = gates.melt(id_vars=["threshold"], value_vars=["coverage", "accuracy"], var_name="metric", value_name="value")
    return _altair_theme(
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="Act only if confidence ≥ threshold", scale=alt.Scale(domain=[0.65, 1.0])),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0.2, 1.0]), axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["coverage", "accuracy"], range=[AMBER, CYAN]), legend=alt.Legend(title=None)),
            tooltip=["threshold", "metric", alt.Tooltip("value:Q", format=".1%")],
        )
        .properties(height=260, title="Coverage falls as the gate hardens; accuracy among the remaining rows rises")
    )


def _comparison_accuracy_chart(table: pd.DataFrame) -> alt.Chart:
    d = table.reset_index().rename(columns={"": "provider"})
    return _altair_theme(
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("provider:N", title=None),
            y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("provider:N", scale=alt.Scale(range=[CYAN, AMBER, TEAL, ROSE]), legend=None),
            tooltip=["provider", alt.Tooltip("Accuracy:Q", format=".1%"), "n", "Accuracy 95% CI"],
        )
        .properties(height=280)
    )


# ---------------------------------------------------------------------------
# Rendering -- Streamlit calls live here; kept thin, delegating to the data
# functions above wherever there's real logic worth testing in isolation.
# ---------------------------------------------------------------------------


def render_overview(df: pd.DataFrame | None, fanout_df: pd.DataFrame | None = None) -> None:
    st.header("Overview")
    st.markdown(
        '<p class="hero-sub">Is Jev just a fast classifier, or is typed probabilistic judgment actually useful as a '
        "programming primitive between deterministic code and generative LLMs? This dashboard "
        "summarizes the recorded results from every experiment in <code>src/experiments/</code>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline">'
        '<div class="pipe"><div class="step">1 · TEXT</div><div class="body">Messy user message</div></div>'
        '<div class="pipe"><div class="step">2 · JEV</div><div class="body">Typed Choice / Noul / Score</div></div>'
        '<div class="pipe"><div class="step">3 · DECISION</div><div class="body">Label + confidence</div></div>'
        '<div class="pipe"><div class="step">4 · CODE</div><div class="body">Branch, route, or fall back</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("See the Try It Yourself tab to run TEXT → JEV → TYPED DECISION → CODE live.")

    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return

    bitext = filter_experiment(df, "bitext_hard_choice")
    multilingual = filter_experiment(df, "multilingual")
    scaling = filter_experiment_prefix(df, "choice_scaling")
    bitext_jev = _prefer_jev(bitext) if not bitext.empty else bitext
    multi_jev = _prefer_jev(multilingual) if not multilingual.empty else multilingual
    bitext_m = compute_metrics(bitext_jev) if not bitext.empty else None
    multi_m = compute_metrics(multi_jev) if not multilingual.empty else None
    error_n = int(df["error"].notna().sum()) if "error" in df.columns else 0

    cards = [
        ("Recorded predictions", f"{len(df):,}", "rows in results.parquet, all experiments"),
        ("Hard-choice accuracy", _fmt_pct(bitext_m["accuracy"]) if bitext_m else "—", f"Bitext · {bitext_jev['provider'].iloc[0] if not bitext.empty else '—'} · n={bitext_m['n'] if bitext_m else 0:,} pooled"),
        ("Multilingual accuracy", _fmt_pct(multi_m["accuracy"]) if multi_m else "—", f"MASSIVE 60-way · n={multi_m['n'] if multi_m else 0:,} pooled"),
        ("Schema / infra errors", str(error_n), "API/schema failures across every scored row"),
    ]
    st.markdown(_kpi_html(cards), unsafe_allow_html=True)
    _meaning(
        "<b>What these KPIs are.</b> Accuracy is correct ÷ scored rows (errors excluded from the denominator). "
        "Hard-choice and multilingual headline numbers pool every recorded sample size for that experiment — "
        "dev/small/main on Bitext, every locale on MASSIVE — so they can differ slightly from a single n=1,000 / n=800 slice in the write-up. "
        "n is always shown. Error count is infrastructure, not wrong labels."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            _section("1", "Multilingual accuracy by locale", "Does the same decision survive a change of language?")
            if multilingual.empty:
                st.caption("Run multilingual.py to fill this panel.")
            else:
                _how_to_read(
                    [
                        ("ONE BAR", "one locale, one provider"),
                        ("X AXIS", "accuracy, 0–100% (higher is better)"),
                        ("HELD FIXED", "same aligned utterances, same 60 intent labels"),
                    ]
                )
                st.altair_chart(_locale_accuracy_chart(multilingual), use_container_width=True)
                spread = locale_spread_insight(multilingual)
                if spread:
                    _insight(spread)
                else:
                    _insight("Accuracy per locale on the same underlying utterances. Intervals live in the Multilingual tab's table.")
    with c2:
        with st.container(border=True):
            _section("2", "Choice scaling", "What does a bigger decision space cost you?")
            if scaling.empty:
                st.caption("Run choice_scaling.py to fill this panel.")
            else:
                tidy = choice_scaling_by_provider(scaling)
                _how_to_read(
                    [
                        ("X AXIS", "K = number of candidate intents offered"),
                        ("HELD FIXED", "the same 100 examples at every K"),
                        ("WHY IT'S FAIR", "truth always sits in the K=5 prefix; raising K only adds distractors"),
                    ]
                )
                acc, conf = _choice_scaling_charts(tidy)
                st.altair_chart(acc, use_container_width=True)
                st.altair_chart(conf, use_container_width=True)
                scaling_line = choice_scaling_insight(tidy)
                if scaling_line:
                    _insight(scaling_line)
    with c3:
        with st.container(border=True):
            _section("3", "Parallel fan-out", "Do extra typed judgments cost extra waiting?")
            if fanout_df is None or fanout_df.empty:
                st.caption("Run fanout.py to fill this panel.")
            else:
                summary = aggregate_fanout_results(fanout_df)
                _how_to_read(
                    [
                        ("X AXIS", "N = questions asked in one <code>system_one</code> call"),
                        ("TOP CHART", "total wall-clock — what a user waits"),
                        ("BOTTOM CHART", "total ÷ N — derived, not measured per question"),
                        ("NO ACCURACY HERE", "most fan-out questions have no ground truth"),
                    ]
                )
                total, per_q = _fanout_charts(summary)
                st.altair_chart(total, use_container_width=True)
                st.altair_chart(per_q, use_container_width=True)
                fan_line = fanout_insight(summary)
                if fan_line:
                    _insight(fan_line, tone="good")

    _section("4", "Primary metrics, by experiment", "Every recorded experiment side by side — accuracy, latency, reliability, cost.")
    _meaning(
        "<b>Accuracy</b> — was the decision right? &nbsp;·&nbsp; <b>p50/p95</b> — typical vs tail latency. &nbsp;·&nbsp; "
        "<b>Error rate</b> — API/schema/reliability, not mislabels. &nbsp;·&nbsp; <b>Cost</b> — token estimate from pricing.py. &nbsp;·&nbsp; "
        "<b># choices</b> — mean candidate-set size (decision complexity). &nbsp;·&nbsp; "
        "<b>Accuracy 95% CI</b> — Wilson interval; overlapping CIs mean you cannot rank those groups."
    )
    _show_table(overview_dashboard(df), reset=True)
    _chips(
        [
            ("Recorded predictions", f"{len(df):,}"),
            ("Experiments", str(df["experiment"].nunique())),
            ("Providers", ", ".join(sorted(df["provider"].unique()))),
            ("Schema / infra errors", str(error_n)),
        ]
    )
    st.metric("Total recorded predictions", len(df))
    _download_button(df, "Download all raw results (CSV)", "results.csv")
    _takeaway(
        "<b>Part I takeaway:</b> Jev is fast, typed, and useful on narrow well-specified choices — but typed does not mean correct, "
        "and confidence is not yet a production guarantee. Each tab below carries the caveat that belongs to its own result."
    )


def render_hard_choices(df: pd.DataFrame | None) -> None:
    st.header("Hard Choices (Bitext)")
    st.caption("5-way choice among semantically close distractors. # choices = 5 for every row here.")
    _meaning(
        "<b>What this chart/table measures.</b> Bitext Customer Support MCQ, test split. Each message is offered five intents; "
        "the four wrong ones are the semantically closest labels to the true intent — not a trivial refund-vs-weather split "
        "(think <code>check_invoice</code> vs <code>get_invoice</code>). A row is correct iff the returned choice equals ground truth. "
        "Runs at different sample sizes (dev 50 / small 250 / main 1,000) share this experiment name and are pooled here."
    )
    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return
    subset = filter_experiment(df, "bitext_hard_choice")
    if subset.empty:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return

    by_provider = compute_metrics_by_group(subset, "provider")
    table = metrics_table(by_provider, mean_choices_by_group(subset, "provider"))
    kpis = []
    for provider, m in by_provider.items():
        kpis.append((provider, _fmt_pct(m["accuracy"]), f"n={m['n']:,} · p50 { _fmt_num(m['p50_latency_ms'])} ms · CI {_format_ci(m['accuracy_ci_low'], m['accuracy_ci_high'])}"))
    st.markdown(_kpi_html(kpis[:4] or [("no provider", "—", "")]), unsafe_allow_html=True)

    left, right = st.columns((1, 1))
    with left:
        with st.container(border=True):
            _section("1", "Accuracy by provider", "Who picks the right intent more often on the same five candidates?")
            _how_to_read(
                [
                    ("ONE BAR", "one provider, all Bitext runs pooled"),
                    ("Y AXIS", "accuracy on scored rows, 0–100%"),
                    ("HOVER", "Wilson 95% CI and n"),
                    ("NOT SHOWN", "errored rows — they leave the denominator"),
                ]
            )
            st.altair_chart(_comparison_accuracy_chart(table), use_container_width=True)
            _insight(
                "Non-overlapping 95% intervals mean the gap is not sampling noise; overlapping ones mean these two providers "
                "are not separable on this slice, however different the bars look."
            )
    with right:
        with st.container(border=True):
            _section("2", "Confidence when right vs. wrong", "Does the reported number carry any information about correctness?")
            _how_to_read(
                [
                    ("BAR PAIR", "same provider, split by outcome"),
                    ("Y AXIS", "mean reported confidence, 0–1"),
                    ("WANTED SHAPE", "cyan (correct) clearly above rose (incorrect)"),
                    ("WHAT IT IS NOT", "calibration — no model is fit here"),
                ]
            )
            conf_rows = []
            for provider, m in by_provider.items():
                conf_rows.append({"provider": provider, "when": "correct", "confidence": m["confidence_when_correct"]})
                conf_rows.append({"provider": provider, "when": "incorrect", "confidence": m["confidence_when_incorrect"]})
            conf_df = pd.DataFrame(conf_rows)
            st.altair_chart(
                _altair_theme(
                    alt.Chart(conf_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("provider:N", title=None),
                        y=alt.Y("confidence:Q", title="Mean confidence", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("when:N", scale=alt.Scale(domain=["correct", "incorrect"], range=[CYAN, ROSE])),
                        xOffset="when:N",
                        tooltip=["provider", "when", alt.Tooltip("confidence:Q", format=".3f")],
                    )
                    .properties(height=280)
                ),
                use_container_width=True,
            )
            gaps = [
                f"{provider} {m['confidence_when_correct']:.2f} vs {m['confidence_when_incorrect']:.2f}"
                for provider, m in by_provider.items()
                if m.get("confidence_when_correct") is not None and m.get("confidence_when_incorrect") is not None
            ]
            if gaps:
                _insight(
                    f"Separation (correct vs. incorrect): {' · '.join(gaps)}. A gap this direction means confidence is a "
                    "<b>useful signal</b>; it does not mean a high number is a guarantee — see the Confidence tab."
                )

    _section("3", "Full metrics", "Everything the two bar charts above leave out.")
    _meaning(
        "One row per provider: accuracy and its Wilson interval, p50/p95 latency, error rate, estimated cost, the confidence split, "
        "mean candidate-set size (5 throughout this experiment), and n."
    )
    _show_table(table, reset=True)

    misses = subset[subset["correct"].eq(False) & subset["text"].notna()]
    if not misses.empty:
        _section("4", "Recorded misses (sample)", "The actual wrong answers, not an aggregate.")
        _meaning("Wrong labels, not infrastructure errors. Near-synonym swaps at high confidence are the failure mode Experiment E / Part II care about.")
        cols = [c for c in ("provider", "text", "ground_truth", "prediction", "confidence", "latency_ms") if c in misses.columns]
        st.dataframe(misses[cols].head(25), use_container_width=True, hide_index=True)
        confident_misses = misses[misses["confidence"].ge(HIGH_CONFIDENCE_THRESHOLD)]
        _chips(
            [
                ("Misses recorded", f"{len(misses):,}"),
                ("Shown below", f"{min(len(misses), 25)}"),
                (f"Of those, confidence ≥ {HIGH_CONFIDENCE_THRESHOLD:.2f}", f"{len(confident_misses):,}"),
            ]
        )
        _insight(
            "Read a few rows rather than the count: the interesting ones are <code>ground_truth</code> and <code>prediction</code> "
            "that a person would also hesitate between, returned at high confidence.",
            tone="warn",
            icon="!",
        )

    _download_button(subset, "Download raw results (CSV)", "bitext_hard_choice.csv")


def render_multilingual(df: pd.DataFrame | None) -> None:
    st.header("Multilingual (MASSIVE, aligned)")
    st.caption("Same underlying utterance, evaluated across languages — accuracy by locale.")
    _meaning(
        "<b>What this measures.</b> MASSIVE test partition, 60 intent labels. An aligned ID is the same utterance translated in every selected locale, "
        "so the decision is identical and only the input language changes. Default slice is 100 IDs × 8 locales = 800 calls per provider. "
        "A row is correct iff the 60-way choice matches that locale's ground-truth intent."
    )
    if df is None:
        _no_data_message("src/experiments/multilingual.py")
        return
    subset = filter_experiment(df, "multilingual")
    if subset.empty:
        _no_data_message("src/experiments/multilingual.py")
        return

    _chips(
        [
            ("Locales", str(subset["locale"].nunique())),
            ("Intent labels", "60"),
            ("Rows recorded", f"{len(subset):,}"),
            ("Providers", ", ".join(sorted(subset["provider"].unique()))),
        ]
    )

    _section("1", "By provider", "Pooled across every locale — the headline multilingual number.")
    _meaning("Pooled across all locales. Compare accuracy, error rate (schema/API — gpt-4o-mini had a non-zero rate in Part I), p50 latency, and cost. Latency here is 60-way, not 5-way.")
    _how_to_read(
        [
            ("ONE BAR", "one provider, all locales pooled"),
            ("Y AXIS", "accuracy, 0–100%"),
            ("DECISION SIZE", "60 candidate intents on every row"),
            ("WATCH", "error rate in the table — it is reliability, not mislabels"),
        ]
    )
    provider_table = metrics_table(compute_metrics_by_group(subset, "provider"))
    st.altair_chart(_comparison_accuracy_chart(provider_table), use_container_width=True)
    _show_table(provider_table, reset=True)
    _insight(
        "A 60-way choice is the hard end of this dataset. Compare the p50 column across providers too: if accuracy differs but "
        "latency does not, the cheaper decision is the one worth routing through code."
    )

    _section("2", "By language (locale)", "Same utterances, only the input language changes.")
    _meaning(
        "Point estimates can look like a ranking (English high, Tamil low). At n=100 the Wilson interval is ~15 points wide and neighbours overlap. "
        "The top-to-bottom spread can be real; a ranking between adjacent locales is not supported. Latency columns tell you whether language changed waiting time."
    )
    _how_to_read(
        [
            ("ONE BAR", "one locale, one provider"),
            ("X AXIS", "accuracy, 0–100%"),
            ("HELD FIXED", "the same aligned utterance IDs in every locale"),
            ("DO NOT", "rank adjacent bars — their intervals overlap"),
        ]
    )
    locale_table = metrics_table(compute_metrics_by_group(subset, "locale"))
    st.altair_chart(_locale_accuracy_chart(subset), use_container_width=True)
    _show_table(locale_table, reset=True)
    spread = locale_spread_insight(subset)
    if spread:
        _insight(spread, tone="warn", icon="!")

    _section("3", "Cross-language consistency", "Is a hard utterance hard everywhere, or only in some languages?")
    _meaning(
        "<b>How to read this histogram.</b> One bar per aligned utterance: how many of its locales were classified correctly. "
        "8/8 = right in every language. 0/8 = wrong everywhere. A bimodal 8/8 vs 0/8 pile means difficulty is mostly the utterance/intent, not the language. "
        "A flat middle (1–5/8) means errors are more language-dependent. Errored rows count as not-correct."
    )
    _how_to_read(
        [
            ("X AXIS", "how many locales got this utterance right"),
            ("Y AXIS", "count of aligned utterances"),
            ("BIMODAL (ends piled)", "difficulty lives in the intent, not the language"),
            ("FLAT MIDDLE", "difficulty is language-dependent"),
        ]
    )
    consistency = compute_cross_language_consistency(subset)
    st.bar_chart(pd.Series(consistency["histogram"], name="count of aligned utterances"))
    _kept("Pooled across every provider in this experiment — the original view, kept.")
    if subset["provider"].nunique() > 1:
        st.markdown("**Same histogram, split by provider** — pooling can hide that Jev is bimodal while the LLM is flatter.")
        for provider, group in subset.groupby("provider"):
            with st.container(border=True):
                st.markdown(f"**{provider}**")
                per_provider = compute_cross_language_consistency(group)
                st.altair_chart(_consistency_chart(per_provider["histogram"]), use_container_width=True)
                st.caption(f"{len(per_provider['per_id'])} aligned IDs.")
    else:
        st.altair_chart(_consistency_chart(consistency["histogram"]), use_container_width=True)
    _insight(
        "The split-by-provider view is the one to trust: a pooled histogram averages two different failure shapes into a middle "
        "that neither provider actually has."
    )

    _download_button(subset, "Download raw results (CSV)", "multilingual.csv")


def render_choice_scaling(df: pd.DataFrame | None) -> None:
    st.header("Choice Scaling")
    st.caption("What happens to accuracy, confidence, and latency as the candidate-set size (K) grows?")
    _meaning(
        "<b>What is held constant.</b> The same 100 MASSIVE examples at every K. Candidate sets are nested frequency-ranked prefixes of the 60 intents "
        "(K=5 ⊂ 10 ⊂ 25 ⊂ 60). Ground truth always sits inside the K=5 set, so raising K only adds distractors. "
        "If accuracy falls while confidence stays pinned at 0.99, the model is getting worse without noticing. "
        "If both fall together, the confidence signal is at least tracking difficulty. Latency staying flat is the other claim: a bigger switch statement should not wait longer."
    )
    if df is None:
        _no_data_message("src/experiments/choice_scaling.py")
        return
    subset = filter_experiment_prefix(df, "choice_scaling")
    if subset.empty:
        _no_data_message("src/experiments/choice_scaling.py")
        return

    table = choice_scaling_table(subset)
    ks = sorted(table["num_choices"].unique())
    _chips(
        [
            ("K values tested", " · ".join(f"K={int(k)}" for k in ks)),
            ("Examples per K", f"{int(table['n'].max()):,} rows"),
            ("Candidate sets", "nested frequency-ranked prefixes"),
        ]
    )

    _section("1", "Pooled across providers", "The raw K table, every provider mixed — kept as-is.")
    _meaning("This is the original tidy table: one row per K, every provider mixed. Use it as a raw dump. The charts below split by provider so a gpt-4o-mini run cannot silently average with Jev.")
    st.dataframe(
        table.set_index("num_choices"),
        use_container_width=True,
        column_config={
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.1%", help="Correct ÷ scored at this K (providers pooled)."),
            "accuracy_95_ci": st.column_config.TextColumn("accuracy_95_ci", help="Wilson 95% CI. At n=100, Jev vs LLM at K=25 overlap — that crossover is not established."),
            "p50_latency_ms": st.column_config.NumberColumn(format="%.0f"),
            "p95_latency_ms": st.column_config.NumberColumn(format="%.0f"),
            "mean_confidence": st.column_config.NumberColumn(format="%.3f", help="Mean reported confidence at this K, correct and incorrect together."),
            "n": st.column_config.NumberColumn(format="%d"),
        },
    )

    tidy = choice_scaling_by_provider(subset)
    _section("2", "By provider — accuracy and confidence", "The two lines that have to move together for confidence to mean anything.")
    _how_to_read(
        [
            ("X AXIS", "K = number of candidate intents offered"),
            ("LEFT Y", "accuracy, 0–100%"),
            ("RIGHT Y", "mean confidence, 0–1 (correct and incorrect pooled)"),
            ("ONE POINT", "one K for one provider"),
            ("BAD SHAPE", "accuracy falls while confidence stays flat"),
            ("GOOD SHAPE", "both decline together"),
        ]
    )
    acc, conf = _choice_scaling_charts(tidy)
    a, b = st.columns(2)
    with a:
        st.altair_chart(acc, use_container_width=True)
    with b:
        st.altair_chart(conf, use_container_width=True)
    scaling_line = choice_scaling_insight(tidy)
    if scaling_line:
        _insight(scaling_line)
    _callout("Solid finding: Jev accuracy and mean confidence decline in step as K grows. Suggestive only: gpt-4o-mini sitting near 80% at high K — intervals overlap at n=100, and the LLM is spending far more wall-clock and tokens.")

    _section("3", "Latency vs K", "Does a bigger switch statement make the user wait longer?")
    _meaning("p50 is the typical call; p95 is the tail. A flat pair of lines means a 60-way choice does not cost waiting relative to a 5-way choice. That is independent of whether accuracy held.")
    _how_to_read(
        [
            ("X AXIS", "K = number of candidate intents"),
            ("Y AXIS", "latency in ms"),
            ("SOLID vs DASHED", "provider"),
            ("p50 / p95", "typical call vs. the slow tail"),
        ]
    )
    st.altair_chart(_latency_chart(tidy), use_container_width=True)
    _insight(
        "Flat lines here and falling lines above are two separate results: widening the decision space costs <b>accuracy</b>, "
        "not <b>time</b>. Only the second one is free."
    )
    st.line_chart(table.set_index("num_choices")[["accuracy"]])
    st.line_chart(table.set_index("num_choices")[["p50_latency_ms", "p95_latency_ms"]])
    _kept("The two line charts above are the original pooled-K views (providers mixed). Kept so nothing previously visible is dropped.")

    _section("4", "Full K × provider table", "Every cell behind the charts, including intervals and n.")
    st.dataframe(
        tidy,
        use_container_width=True,
        hide_index=True,
        column_config={
            "accuracy": st.column_config.NumberColumn(format="%.1%"),
            "mean_confidence": st.column_config.NumberColumn(format="%.3f"),
            "p50_latency_ms": st.column_config.NumberColumn(format="%.0f"),
            "p95_latency_ms": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    _download_button(subset, "Download raw results (CSV)", "choice_scaling.csv")


def render_parallel_decisions(fanout_df: pd.DataFrame | None) -> None:
    st.header("Parallel Decisions (Fan-out)")
    st.caption("One message, many simultaneous questions in one call. Latency behavior only — most questions have no ground truth, so no accuracy is shown here.")
    _meaning(
        "<b>What this experiment is.</b> 5 Bitext messages × question counts {1, 5, 10, 25, 50}, one <code>system_one</code> call per pair. "
        "Questions are a prefix of a fixed list (intent Choice, then Noul/Score triage flags like “is this urgent”). "
        "<b>Total latency</b> is wall-clock for the whole call — what a user waits. "
        "<b>Per-question latency</b> is total ÷ N, a derived metric: if total is flat, this line collapses. "
        "Do not read it as “each extra question added 4.6 ms of sequential work.”"
    )
    if fanout_df is None:
        _no_data_message("src/experiments/fanout.py")
        return
    summary = aggregate_fanout_results(fanout_df)
    _chips(
        [
            ("Question counts", " · ".join(str(int(n)) for n in sorted(summary["num_questions"].unique()))),
            ("Messages per count", str(fanout_df["message_id"].nunique())),
            ("Metric", "wall-clock latency only"),
        ]
    )

    _section("1", "Summary table", "The means the two charts below are drawn from.")
    _meaning("Means across the 5 messages. input_tokens / output_tokens are why this is not a cost result: tokens rise with N even when milliseconds do not.")
    st.dataframe(summary.set_index("num_questions") if "num_questions" in summary.columns else summary, use_container_width=True)

    _section("2", "Total vs. per-question latency", "One measured curve and one derived from it — not two findings.")
    _how_to_read(
        [
            ("X AXIS (both)", "N = questions asked in one call"),
            ("LEFT CHART", "total wall-clock in ms — <b>measured</b>"),
            ("RIGHT CHART", "total ÷ N — <b>derived</b>"),
            ("THE CLAIM", "flat total ⇒ extra judgments are nearly free in time"),
        ]
    )
    total, per_q = _fanout_charts(summary)
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(total, use_container_width=True)
        _meaning("If this line is nearly flat from 1 → 50, a triage step that wants twenty typed signals costs about the same wait as one.")
    with c2:
        st.altair_chart(per_q, use_container_width=True)
        _meaning("This is total ÷ N. A collapse of ~35× is arithmetic on a flat total, not a second independent miracle.")
    fan_line = fanout_insight(summary)
    if fan_line:
        _insight(fan_line, tone="good")

    st.line_chart(summary.set_index("num_questions")[["total_latency_ms", "latency_per_question_ms"]])
    _kept("Original Streamlit line chart kept. Mixing total and per-question on one axis squeezes the per-question series — prefer the two labeled charts above.")

    if {"input_tokens", "output_tokens"}.issubset(summary.columns):
        _section("3", "Tokens vs. question count", "The part of fan-out that is not free.")
        _how_to_read(
            [
                ("X AXIS", "N = simultaneous questions"),
                ("Y AXIS", "mean tokens per call"),
                ("TWO SERIES", "input tokens and output tokens"),
                ("WHY IT MATTERS", "tokens are billed; milliseconds are what stayed flat"),
            ]
        )
        st.altair_chart(_token_chart(summary), use_container_width=True)
        _insight(
            "This chart is the reason the fan-out result is stated as <b>latency, not cost</b>. Tokens climb with N even where "
            "wall-clock does not.",
            tone="warn",
            icon="!",
        )
        _callout("Say latency, not cost. For Jev in Part I, input tokens went ~350 → 1,260 and output ~82 → 1,017 from 1 to 50 questions. You pay for those. Waiting is what is nearly free.")

    _download_button(fanout_df, "Download raw results (CSV)", "fanout_results.csv")


def render_jev_vs_llm(df: pd.DataFrame | None) -> None:
    st.header("Jev vs LLM")
    st.caption("Every column is headed by an explicit model name — never a generic \"LLM\".")
    _meaning(
        "<b>Fairness.</b> Both providers implement <code>predict(example) → PredictionResult</code>, same examples, same Choice contract, same instructions. "
        "Only the judgment differs. gpt-4o-mini is the cheap 2024 analogue this account could call (gpt-6-luna returned HTTP 403) — the right role comparison, not a current-gen bake-off."
    )
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
    table = metrics_table(compute_metrics_by_group(subset, "provider"), mean_choices_by_group(subset, "provider"))
    _chips(
        [
            ("Experiment", chosen),
            ("Providers compared", str(len(providers))),
            ("Rows", f"{len(subset):,}"),
            ("Candidate set", f"{subset['candidates'].apply(len).mean():.0f}-way"),
        ]
    )
    if not table.empty:
        _section("1", "Accuracy, head to head", "Same examples, same contract — only the judgment differs.")
        _how_to_read(
            [
                ("ONE BAR", "one named model"),
                ("Y AXIS", "accuracy, 0–100%"),
                ("HOVER", "Wilson 95% CI and n"),
                ("READ WITH", "the latency and cost columns below — accuracy alone is half the comparison"),
            ]
        )
        st.altair_chart(_comparison_accuracy_chart(table), use_container_width=True)
        _meaning("Accuracy bars for whoever is recorded on this experiment. The table below still has latency, error rate, cost, confidence split, # choices, n, and CIs — none of that is dropped.")
        _insight(
            "The honest comparison is the whole row, not the bar: a model can win on accuracy and lose the routing decision "
            "on p95 latency, error rate, or cost per call."
        )
    _section("2", "Full comparison table", "Accuracy, intervals, latency, reliability, cost, confidence split, n.")
    _show_table(table, reset=True)
    _download_button(subset, "Download raw results (CSV)", f"{chosen}_by_provider.csv")


def render_confidence(df: pd.DataFrame | None) -> None:
    st.header("Confidence")
    st.caption("Exploratory only (Part I) — not calibration. Can we trust this confidence? Part II's question.")
    _meaning(
        "<b>What “confidence” is here.</b> The probability Jev/the adapter attached to the chosen label, recorded on each row. "
        "This tab is descriptive: it does not fit a calibration model, does not pick a production threshold, and does not validate on a held-out set. "
        "A useful signal would (1) sit higher on correct rows than wrong ones, and (2) let you gate: only act above a cutoff, route the rest elsewhere."
    )
    if df is None:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return
    scored = df[df["correct"].notna() & df["confidence"].notna()]
    if scored.empty:
        _no_data_message("src/experiments/bitext_hard_choice.py")
        return

    experiments = ["(all scored rows)", *sorted(scored["experiment"].unique())]
    providers = ["(all providers)", *sorted(scored["provider"].unique())]
    f1, f2 = st.columns(2)
    with f1:
        experiment_choice = st.selectbox("Experiment filter", experiments)
    with f2:
        provider_choice = st.selectbox("Provider filter", providers)
    if experiment_choice != "(all scored rows)":
        scored = scored[scored["experiment"] == experiment_choice]
    if provider_choice != "(all providers)":
        scored = scored[scored["provider"] == provider_choice]
    if scored.empty:
        st.warning("No scored rows for that filter. The default “(all scored rows)” view is still available.")
        return

    dist = confidence_distribution_by_correctness(scored)
    n_correct = int(scored["correct"].eq(True).sum())
    n_wrong = int(scored["correct"].eq(False).sum())
    mean_ok = float(scored.loc[scored["correct"].eq(True), "confidence"].mean()) if n_correct else None
    mean_bad = float(scored.loc[scored["correct"].eq(False), "confidence"].mean()) if n_wrong else None
    confidently_wrong = scored[scored["correct"].eq(False) & (scored["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)]
    st.markdown(
        _kpi_html(
            [
                ("Mean conf · correct", _fmt_num(mean_ok, 3), f"n={n_correct:,} scored correct"),
                ("Mean conf · wrong", _fmt_num(mean_bad, 3), f"n={n_wrong:,} scored wrong"),
                ("Confidently wrong", f"{len(confidently_wrong) / len(scored):.1%}" if len(scored) else "—", f"incorrect and confidence ≥ {HIGH_CONFIDENCE_THRESHOLD:.2f}"),
                ("Scored rows", f"{len(scored):,}", f"{experiment_choice} · {provider_choice}"),
            ]
        ),
        unsafe_allow_html=True,
    )
    _callout(
        f"Confident-and-wrong is the production-visible failure: those rows would pass a {HIGH_CONFIDENCE_THRESHOLD:.2f} gate. "
        "Unsure-and-wrong can be routed to an LLM or a human. Part II asks whether this rate holds under ambiguity, distractors, and OOD."
    )

    _section("1", "Confidence distribution: correct vs. incorrect", "Do the two outcome classes actually sit at different confidence levels?")
    _meaning("Each row of the table is one correctness class. n / mean / median / std / min / max of the recorded confidence. Real separation = mean(correct) ≫ mean(incorrect). Both models in Part I showed this; they were also confidently wrong at a similar rate (~2.3–2.5% on Bitext n=1,000).")
    _how_to_read(
        [
            ("ONE ROW", "all correct rows, or all incorrect rows"),
            ("THE COLUMN THAT MATTERS", "mean — the gap between the two"),
            ("std / min", "how much the classes overlap in practice"),
            ("WHAT IT IS NOT", "calibration — no held-out validation here"),
        ]
    )
    st.dataframe(dist.set_index("correct"), use_container_width=True)
    if mean_ok is not None and mean_bad is not None:
        _insight(
            f"Mean confidence is <b>{mean_ok:.3f}</b> when correct and <b>{mean_bad:.3f}</b> when wrong — a "
            f"{(mean_ok - mean_bad):.3f} gap. That is a usable signal, but the two distributions still overlap: "
            "the min on the correct row and the max on the incorrect row tell you by how much."
        )

    histogram = confidence_histogram(scored).set_index("confidence_bin").rename(columns={False: "Incorrect", True: "Correct"})
    _section("2", "Histogram", "Where the predictions actually pile up, and what is hiding in the top bin.")
    _meaning(
        "X-axis bins are confidence intervals (see experiments/confidence.py: 0–0.5, 0.5–0.6, …, 0.95–1.0). "
        "Height is a count of predictions, split by correctness. A pile-up in 0.95–1.0 that is almost all Correct is the gate working; "
        "any Incorrect mass in that last bin is the 2.3% problem. This is a count chart, not a density — more correct rows exist, so blue will dominate."
    )
    _how_to_read(
        [
            ("X AXIS", "confidence bin (right-inclusive except the first)"),
            ("Y AXIS", "count of predictions — <b>not</b> a density"),
            ("TWO SERIES", "cyan = correct, rose = incorrect"),
            ("THE BIN TO STUDY", "0.95–1.0 — any rose there is a confidently wrong call"),
        ]
    )
    st.altair_chart(_confidence_hist_chart(histogram), use_container_width=True)
    top_bin = histogram.tail(1)
    if not top_bin.empty and "Incorrect" in top_bin.columns:
        n_top_wrong = int(top_bin["Incorrect"].iloc[0])
        n_top_right = int(top_bin["Correct"].iloc[0]) if "Correct" in top_bin.columns else 0
        if n_top_right + n_top_wrong:
            _insight(
                f"In the top bin ({top_bin.index[0]}): {n_top_right:,} correct and <b>{n_top_wrong:,} incorrect</b> — "
                f"{n_top_wrong / (n_top_right + n_top_wrong):.1%} of the most-confident predictions are wrong. Because this is a "
                "count chart and correct rows dominate overall, that slice is easy to miss by eye.",
                tone="warn",
                icon="!",
            )
    st.bar_chart(histogram[["Incorrect", "Correct"]])
    _kept("Original stacked/grouped bar_chart kept underneath the labeled Altair version.")

    _section("3", "Would thresholding on confidence work as a trust gate?", "The practical question: act automatically above T, route the rest.")
    st.caption("Coverage = fraction of all predictions clearing that threshold. Still descriptive, not calibration.")
    _meaning(
        "<b>Coverage</b> = share of all scored rows with confidence ≥ T (including the wrong ones that sneak through). "
        "<b>Accuracy</b> = among only those rows, how often were they correct. "
        "A practical router wants high coverage and high accuracy: Jev at T=0.90 on Bitext n=1,000 was ~86% coverage @ ~97.9% accuracy; "
        "gpt-4o-mini at the same T kept only ~40% of traffic. Raising T always trades coverage for accuracy on this plot."
    )
    _how_to_read(
        [
            ("X AXIS", "T = the cutoff you would act above"),
            ("AMBER LINE", "coverage — share of traffic you keep"),
            ("CYAN LINE", "accuracy among only the kept rows"),
            ("THE TRADE", "raising T always buys accuracy with coverage"),
            ("n_wrong COLUMN", "wrong predictions that still execute at that T"),
        ]
    )
    gates = accuracy_at_thresholds(scored)
    st.altair_chart(_gate_chart(gates), use_container_width=True)
    gate_line = confidence_gate_insight(gates)
    if gate_line:
        _insight(gate_line, tone="warn", icon="!")
    st.dataframe(
        gates.set_index("threshold"),
        use_container_width=True,
        column_config={
            "coverage": st.column_config.NumberColumn(format="%.1%", help="Fraction of all scored predictions with confidence ≥ threshold."),
            "accuracy": st.column_config.NumberColumn(format="%.1%", help="Accuracy among rows that cleared the threshold."),
            "n": st.column_config.NumberColumn(format="%d"),
            "n_correct": st.column_config.NumberColumn(format="%d"),
            "n_wrong": st.column_config.NumberColumn(format="%d", help="These still execute if you gate at this threshold."),
        },
    )

    _section("4", "Example predictions by quadrant", "The four cases a router has to handle, with real rows in each.")
    _meaning(
        f"High = confidence ≥ {HIGH_CONFIDENCE_THRESHOLD:.2f} (the routing-demo default). Low = confidence &lt; {LOW_CONFIDENCE_THRESHOLD:.2f}. "
        "Up to 3 deterministically sampled rows per quadrant (seed 42). High-confidence wrong is the row you would have auto-executed."
    )
    _how_to_read(
        [
            ("HIGH · CORRECT", "auto-execute — the case the gate exists for"),
            ("HIGH · WRONG", "auto-executed and wrong — the production failure"),
            ("LOW · CORRECT", "escalated unnecessarily — the cost of the gate"),
            ("LOW · WRONG", "caught and routed away — the gate working"),
        ]
    )
    quadrants = quadrant_examples(scored, high=HIGH_CONFIDENCE_THRESHOLD, low=LOW_CONFIDENCE_THRESHOLD)
    for name, rows in quadrants.items():
        with st.expander(f"{name.replace('_', ' ').title()} ({len(rows)})"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
    _insight(
        "Open <b>High Confidence Wrong</b> first. Those rows passed every check this dashboard can apply and would still have "
        "executed the wrong branch — which is why Part I calls confidence a signal and not a guarantee.",
        tone="warn",
        icon="!",
    )
    _download_button(scored, "Download scored results (CSV)", "confidence.csv")
    _takeaway(
        "<b>Confidence tab caveat:</b> everything here is descriptive. No calibration model is fit, no threshold is validated on a "
        "held-out set, and the confidently-wrong rate is measured on this data only — not promised under ambiguity or distribution shift."
    )


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
    st.markdown(
        '<div class="pipeline">'
        '<div class="pipe"><div class="step">TEXT</div><div class="body">Your message, unparsed</div></div>'
        '<div class="pipe"><div class="step">JEV</div><div class="body">One system_one call</div></div>'
        '<div class="pipe"><div class="step">TYPED DECISION</div><div class="body">A label from the contract + confidence</div></div>'
        '<div class="pipe"><div class="step">CODE</div><div class="body">execute_route(intent) or N answers to branch on</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    _meaning(
        "This is a live call against your <code>TYPESAFE_API_KEY</code>, not a replay of the Parquet. "
        "Customer-support mode uses the five routing-demo intents. MASSIVE mode uses all 60 intents (the hard end of Experiment C). "
        "Parallel mode fans out 1–50 questions about one message (Experiment D) — latency only, no accuracy."
    )
    _how_to_read(
        [
            ("CUSTOMER-SUPPORT CHOICES", "5 candidates — the easy end, matching the routing demo"),
            ("MASSIVE 60-INTENT", "60 candidates — the hard end of the Choice Scaling tab"),
            ("PARALLEL JUDGMENTS", "N typed questions in one call — reproduces the fan-out result live"),
            ("NO GROUND TRUTH", "nothing you type is scored; there is no accuracy to report here"),
        ]
    )

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
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-kicker">Part I · benchmarking accuracy, confidence, and scalability</div>', unsafe_allow_html=True)
    st.title("jev-decision-lab")
    st.markdown('<p class="hero-sub">Typed probabilistic decisions as a programming primitive — every number below is computed from the recorded Parquet, not copied from the write-up.</p>', unsafe_allow_html=True)

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
        render_overview(results_df, fanout_df)
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
