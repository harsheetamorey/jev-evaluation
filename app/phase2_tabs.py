"""Phase II dashboard tabs (rendering only), used by app/streamlit_app.py's main().

Every tab READS precomputed artifacts through src/phase2/dashboard_data.py. Nothing here calls Jev or an
LLM, and no headline number is hard-coded: each figure is computed from a file on disk. Sample counts are
always shown; simulated metrics say SIMULATED; cost figures say ESTIMATED; Jev and the reference LLM are
labelled distinctly. No tab draws a conclusion the data has not been read for.
"""

from types import SimpleNamespace
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from phase2.dashboard_data import (
    baseline_dir,
    bins_for_group,
    dataset_info,
    filter_failures,
    group_options,
    load_csv,
    load_json,
    load_jsonl,
    load_matched,
    phase2_root,
    provider_label,
    risk_at,
    risk_curve,
    run_cascade_simulation,
    sparse_bin_warnings,
    workloads,
)

PHASE2_TABS = ["Calibration", "Risk / Coverage", "Stability", "Context Stress", "OOD", "Adversarial", "Multilingual Reliability", "Failure Museum", "Cascade Simulator"]
LIVE_RUN_NOTE = "Stress-experiment results appear here after their live evaluation has been run and analysed (the frozen datasets are built, the live runs are a separate approved step)."


def _pct(v: Any, digits: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{100 * float(v):.{digits}f}%"


def _num(v: Any, digits: int = 3) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):.{digits}f}"


def _table(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch", hide_index=True)


def _missing(ui: SimpleNamespace, what: str, command: str, dataset: str | None = None) -> None:
    ui.no_data(command)
    if dataset and (info := dataset_info(dataset)):
        st.caption(f"Frozen {what} dataset: {info['n_rows']} rows from {info['n_source_examples']} source examples (sha256 {info['sha256'][:12]}...). {LIVE_RUN_NOTE}")


def _provider_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "provider" not in df.columns or df["provider"].nunique() < 2:
        return df
    options = sorted(df["provider"].unique(), key=lambda p: (p != "jev", p))
    choice = st.selectbox("Provider", options, format_func=provider_label, key=key)
    return df[df["provider"] == choice]


# --- 1. Calibration ------------------------------------------------------------------------------


def _reliability_chart(gb: pd.DataFrame) -> alt.Chart:
    pop = gb[gb["sample_count"] > 0]
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#94a3b8").encode(x="x:Q", y="y:Q")
    pts = (
        alt.Chart(pop)
        .mark_circle(color="#22d3ee", opacity=0.85)
        .encode(
            x=alt.X("mean_confidence:Q", title="Mean reported confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("actual_accuracy:Q", title="Actual accuracy in bin", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("sample_count:Q", title="n in bin", scale=alt.Scale(range=[40, 900])),
            tooltip=["bin_label", "sample_count", alt.Tooltip("mean_confidence:Q", format=".3f"), alt.Tooltip("actual_accuracy:Q", format=".3f"), alt.Tooltip("calibration_gap:Q", format=".3f")],
        )
    )
    return (diag + pts).properties(height=340, title="Reliability diagram (dashed line = perfect agreement)")


def render_calibration(ui: SimpleNamespace) -> None:
    ui.section("P2·1", "Calibration", "Does reported confidence correspond to actual correctness?")
    root = phase2_root() / "calibration"
    bins, summary = load_csv(root / "calibration_bins.csv"), load_csv(root / "calibration_summary.csv")
    if bins is None or summary is None:
        _missing(ui, "calibration", "src/phase2/run_calibration.py")
        return
    levels = list(summary["level"].unique())
    level = st.selectbox("Analysis level", levels, index=levels.index("provider_dataset_experiment") if "provider_dataset_experiment" in levels else 0, key="p2_cal_level")
    options = group_options(summary, level)
    label = st.selectbox("Group", list(options), key="p2_cal_group")
    row = options[label]
    if bool(row["pooled_across_workloads"]):
        ui.callout("POOLED across different workloads. Use a per-experiment level to avoid mixing unlike tasks.")
    gb = bins_for_group(bins, row)
    st.markdown(ui.kpi_html([("n (labeled predictions)", f"{int(row['n'])}", f"{int(row['n_distinct_examples'])} distinct examples"), ("Accuracy", _pct(row["accuracy"]), "share correct"), ("ECE", _num(row["ece"]), "10 equal-width bins, weighted by n"),
                             ("Brier (correctness)", _num(row["brier_correctness"]), "binary, NOT multiclass"), ("Mean conf. · correct", _num(row["mean_confidence_correct"]), f"n={int(row['n_correct'])}"), ("Mean conf. · wrong", _num(row["mean_confidence_incorrect"]), f"n={int(row['n_incorrect'])}")]), unsafe_allow_html=True)
    if row["provider"] != "jev":
        ui.callout("Reference LLM, shown for provenance. Its confidence semantics have not been investigated.")
    for w in [f"Sparse bin (n < 30): {x}" for x in sparse_bin_warnings(gb)]:
        st.warning(w)
    st.altair_chart(ui.altair_theme(_reliability_chart(gb)), width="stretch")
    st.caption("Confidence is the probability of the CHOSEN label only, stored to 2 decimals. Accuracy and calibration are different properties. Read every bin together with its n; empty bins are kept in the table.")
    _table(gb[["bin_label", "sample_count", "mean_confidence", "actual_accuracy", "calibration_gap"]])


# --- 2. Risk / coverage --------------------------------------------------------------------------


def render_risk_coverage(ui: SimpleNamespace) -> None:
    ui.section("P2·2", "Risk / Coverage", "If we reject uncertain predictions, how much accuracy do we gain and how much automation do we lose?")
    table = load_csv(phase2_root() / "selective_prediction" / "risk_coverage.csv")
    if table is None:
        _missing(ui, "risk-coverage", "src/phase2/run_selective.py")
        return
    levels = list(table["level"].unique())
    level = st.selectbox("Analysis level", levels, index=levels.index("provider_dataset_experiment") if "provider_dataset_experiment" in levels else 0, key="p2_rc_level")
    firsts = table[(table["level"] == level) & (table["threshold"] == table["threshold"].min())]
    options = group_options(firsts, level)
    label = st.selectbox("Group", list(options), key="p2_rc_group")
    key = {c: options[label].get(c) for c in ("run_id", "experiment", "provider", "dataset", "num_choices", "locale")}
    curve = risk_curve(table, key, level)
    if bool(curve["pooled_across_workloads"].iloc[0]):
        ui.callout("POOLED across different workloads.")
    lo, hi = float(curve["threshold"].min()), float(curve["threshold"].max())
    t = st.slider("Confidence threshold (accept if confidence ≥ threshold)", lo, hi, 0.90, 0.01, key="p2_rc_threshold")
    r = risk_at(curve, round(t, 2))
    st.markdown(ui.kpi_html([("Coverage (accepted / total)", _pct(r["coverage"]), f"{r['accepted']} of {r['total']} accepted"), ("Accepted accuracy", _pct(r["accepted_accuracy"]), "n/a when nothing is accepted"), ("Risk (accepted error rate)", _pct(r["risk"]), f"{r['rejected']} rejected → fallback")]), unsafe_allow_html=True)
    plot = curve.dropna(subset=["risk"])
    line = alt.Chart(plot).mark_line(color="#22d3ee", point=True).encode(x=alt.X("coverage:Q", title="Coverage", scale=alt.Scale(domain=[0, 1])), y=alt.Y("risk:Q", title="Risk (error rate among accepted)"), tooltip=["threshold", "accepted", "coverage", "risk"])
    now = alt.Chart(plot[(plot["threshold"] - round(t, 2)).abs() < 1e-9]).mark_point(color="#fbbf24", size=220, filled=True).encode(x="coverage:Q", y="risk:Q")
    st.altair_chart(ui.altair_theme((line + now).properties(height=320, title="Risk–coverage curve (amber = current threshold)")), width="stretch")
    st.caption("Accepted accuracy need not rise with the threshold; the curve shows what the data does. No operating point is recommended here.")
    _table(curve[curve["is_standard_threshold"]][["threshold", "total", "accepted", "coverage", "accepted_accuracy", "risk"]])


# --- 3-6. Stress experiment tabs (read analysed live results) ------------------------------------


def render_stability(ui: SimpleNamespace) -> None:
    ui.section("P2·3", "Stability", "Same decision when the wording changes (paraphrase / surface) or the text gets noisy?")
    stab, noise = phase2_root() / "stability", phase2_root() / "noise"
    fam, typ = load_csv(stab / "stability_by_family.csv"), load_csv(stab / "stability_by_variant_type.csv")
    st.markdown("#### Paraphrase and surface stability (paired with each source's original)")
    if fam is None:
        _missing(ui, "stability", "src/phase2/run_stability.py analyze", "stability")
    else:
        _table(_provider_filter(fam, "p2_stab_provider")[["provider", "variant_family", "n_pairs", "label_agreement", "decision_flip_rate", "accuracy_base", "accuracy_variant", "accuracy_delta", "mean_confidence_delta", "mean_abs_confidence_delta"]])
        if typ is not None:
            _table(_provider_filter(typ, "p2_stab_type_provider")[["provider", "variant_type", "n_pairs", "decision_flip_rate", "accuracy_delta", "mean_abs_confidence_delta"]])
    st.markdown("#### Noise robustness by severity (semantic integrity valid only; questionable rows are reported separately)")
    sev = load_csv(noise / "noise_valid_only_by_severity.csv")
    if sev is None:
        _missing(ui, "noise", "src/phase2/run_noise.py analyze", "noise")
    else:
        _table(_provider_filter(sev, "p2_noise_provider")[["provider", "severity", "n_pairs", "decision_flip_rate", "accuracy_base", "accuracy_variant", "accuracy_drop", "mean_confidence_delta"]])
        by_type = load_csv(noise / "noise_by_type.csv")
        if by_type is not None:
            _table(_provider_filter(by_type, "p2_noise_type_provider")[["provider", "noise_type", "semantic_integrity", "n_pairs", "decision_flip_rate", "accuracy_drop"]])


def render_context_stress(ui: SimpleNamespace) -> None:
    ui.section("P2·4", "Context Stress", "Does irrelevant state change the decision, and does relevant context help?")
    pol = load_csv(phase2_root() / "context_pollution" / "context_by_level.csv")
    st.markdown("#### Irrelevant context by size (paired with the no-context version)")
    if pol is None:
        _missing(ui, "context-pollution", "src/phase2/run_context_pollution.py analyze", "context_pollution")
    else:
        _table(_provider_filter(pol, "p2_ctx_provider")[["provider", "context_level", "n_pairs", "accuracy_delta", "decision_flip_rate", "mean_confidence_delta", "mean_latency_delta_ms", "mean_input_tokens_variant"]])
    st.markdown("#### Relevant vs irrelevant context (paired triplets)")
    cond, trans = load_csv(phase2_root() / "context" / "relevance_by_condition.csv"), load_csv(phase2_root() / "context" / "relevance_transitions.csv")
    if cond is None:
        _missing(ui, "context-relevance", "src/phase2/run_context_relevance.py analyze", "context_relevance")
    else:
        _table(_provider_filter(cond, "p2_rel_provider"))
        if trans is not None:
            st.caption("Transitions read from the first-named condition to the second: wrong_to_correct means the base answer was wrong and the compared answer is right.")
            _table(_provider_filter(trans, "p2_trans_provider"))


def render_ood(ui: SimpleNamespace) -> None:
    ui.section("P2·5", "OOD", "What happens when none of the choices is correct, with and without an explicit fallback?")
    root = phase2_root() / "ood"
    forced, fallback = load_csv(root / "ood_forced_choice.csv"), load_csv(root / "ood_fallback.csv")
    if forced is None or fallback is None:
        _missing(ui, "OOD", "src/phase2/run_ood.py analyze", "ood")
        return
    ui.callout("Choosing 'other' is not evidence that Jev detects OOD; this tests whether an explicit fallback helps when the taxonomy lacks the answer.")
    st.markdown("#### Forced choice: confidence and false-confidence (share of predictions at or above each threshold)")
    _table(_provider_filter(forced, "p2_ood_forced_provider"))
    st.markdown("#### Explicit fallback: 'other' selection")
    _table(_provider_filter(fallback, "p2_ood_fallback_provider"))


def render_adversarial(ui: SimpleNamespace) -> None:
    ui.section("P2·6", "Adversarial", "Can untrusted text in the state steer the typed decision?")
    root = phase2_root() / "adversarial"
    by_attack = load_csv(root / "adversarial_by_attack.csv")
    if by_attack is None:
        _missing(ui, "adversarial", "src/phase2/run_adversarial.py analyze", "adversarial")
        return
    st.caption("attack_success = attacked prediction is the attacker's target AND the clean prediction was not.")
    _table(_provider_filter(by_attack, "p2_adv_provider")[["provider", "attack_type", "n_pairs", "attack_success_rate", "attack_success_rate_among_attackable", "decision_flip_rate", "target_label_selection_rate", "clean_accuracy", "attacked_accuracy", "accuracy_drop", "mean_confidence_delta"]])
    pairs = load_csv(root / "adversarial_pairs.csv")
    if pairs is not None:
        st.markdown("#### Attacked examples")
        only = st.checkbox("Only successful attacks", key="p2_adv_only_success")
        shown = pairs[pairs["attack_success"] == True] if only else pairs  # noqa: E712
        st.caption(f"{len(shown)} of {len(pairs)} attacked pairs")
        _table(shown.head(200))


# --- 7. Multilingual reliability -----------------------------------------------------------------


def render_multilingual_reliability(ui: SimpleNamespace) -> None:
    ui.section("P2·7", "Multilingual Reliability", "Are failures language-specific, intent-specific, or shared across languages?")
    root = phase2_root() / "multilingual"
    summary, lang, aligned = load_json(root / "consistency_summary.json"), load_csv(root / "language_reliability.csv"), load_csv(root / "aligned_reliability.csv")
    if summary is None or lang is None or aligned is None:
        _missing(ui, "multilingual", "src/phase2/run_multilingual.py")
        return
    pop = summary["population"]
    st.caption(f"{pop['n_aligned_source_ids']} aligned source IDs · {len(pop['locales_found'])} locales · {pop['n_intents']} intents. Repeated measurements collapsed to the latest ({pop['repeated_measurement_rows_removed']} rows).")
    provider = st.selectbox("Provider", sorted(summary["providers"], key=lambda p: (p != "jev", p)), format_func=provider_label, key="p2_ml_provider")
    ps = summary["providers"][provider]
    st.markdown(ui.kpi_html([("Aligned examples", str(ps["n_aligned_examples"]), f"{ps['n_with_all_locales_scored']} with all locales scored"), ("Correct in all locales", str(ps["all_8_correct"]), "cross-language consistent"), ("Wrong in all locales", str(ps["all_8_wrong"]), "shared failure"),
                             ("Predictions differ across locales", str(ps["n_prediction_unstable"]), "unstable label choice")]), unsafe_allow_html=True)
    dist = pd.DataFrame({"locales_correct": list(ps["n_correct_distribution"]), "examples": list(ps["n_correct_distribution"].values())})
    chart = alt.Chart(dist).mark_bar(color="#22d3ee").encode(x=alt.X("locales_correct:O", title="Number of locales correct (per aligned example)"), y=alt.Y("examples:Q", title="Aligned examples"), tooltip=["locales_correct", "examples"]).properties(height=260, title="Cross-language consistency")
    st.altair_chart(ui.altair_theme(chart), width="stretch")
    st.markdown("#### How failures are shaped (deterministic classes from correctness pattern)")
    _table(pd.DataFrame({"class": list(ps["classification_counts"]), "aligned_examples": list(ps["classification_counts"].values())}))
    st.markdown("#### Per-language reliability (listed by locale code, not ranked; intervals overlap heavily at this n)")
    _table(lang[lang["provider"] == provider][["locale", "n_scored", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "mean_confidence", "n_wrong_here_only", "n_wrong_here_and_in_most_languages"]])
    st.markdown("#### Aligned examples")
    classes = ["(all)"] + sorted(aligned["classification"].unique())
    pick = st.selectbox("Class", classes, key="p2_ml_class")
    view = aligned[aligned["provider"] == provider]
    view = view if pick == "(all)" else view[view["classification"] == pick]
    st.caption(f"{len(view)} aligned examples")
    _table(view[["source_example_id", "intent", "n_locales_available", "n_correct", "classification", "prediction_unstable", "mean_confidence"]].head(300))


# --- 8. Failure Museum ---------------------------------------------------------------------------


def render_failure_museum(ui: SimpleNamespace) -> None:
    ui.section("P2·8", "Failure Museum", "Every recorded failure, filterable and individually inspectable.")
    df = load_jsonl(phase2_root() / "failures" / "failures.jsonl")
    if df is None:
        _missing(ui, "failure", "src/phase2/run_failures.py")
        return
    st.caption("A failure can carry several tags and no single cause is forced; 'no tags' means the evidence does not classify it. Confidence tags use configurable slicing thresholds, not optimal ones.")
    tag_options = sorted({t for tags in df["failure_tags"] for t in tags})
    c1, c2, c3 = st.columns(3)
    experiments = c1.multiselect("Experiment", sorted(df["experiment"].unique()), key="p2_fm_exp")
    providers = c2.multiselect("Provider", sorted(df["provider"].unique()), format_func=provider_label, key="p2_fm_provider")
    locales = c3.multiselect("Language / locale", sorted(df["locale"].dropna().unique()), key="p2_fm_locale")
    tags = st.multiselect("Failure tag (matches any selected)", tag_options, key="p2_fm_tags")
    conf = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01, key="p2_fm_conf")
    d1, d2 = st.columns(2)
    truths = d1.multiselect("Ground truth", sorted(df["ground_truth"].dropna().unique()), key="p2_fm_truth")
    preds = d2.multiselect("Prediction", sorted(df["prediction"].dropna().unique()), key="p2_fm_pred")
    shown = filter_failures(df, experiments=experiments, providers=providers, locales=locales, tags=tags, conf_range=conf, ground_truth=truths, prediction=preds)
    st.markdown(f"**{len(shown)}** of {len(df)} failures match")
    flat = shown.assign(failure_tags=shown["failure_tags"].map(", ".join))
    _table(flat[["failure_id", "experiment", "provider", "locale", "ground_truth", "prediction", "confidence", "failure_tags", "stress_condition"]].head(500))
    if shown.empty:
        return
    pick = st.selectbox("Inspect a failure", list(shown["failure_id"].head(500)), key="p2_fm_pick")
    r = shown[shown["failure_id"] == pick].iloc[0]
    st.markdown(f"**{provider_label(r['provider'])}** · `{r['experiment']}` · truth **{r['ground_truth']}** → predicted **{r['prediction']}** · confidence {_num(r['confidence'], 2)}")
    st.text_area("Input", r["input"] or "", height=90, disabled=True, key="p2_fm_input")
    if r.get("transformed_input"):
        st.text_area("Transformed input", r["transformed_input"], height=90, disabled=True, key="p2_fm_tinput")
    st.write({"tags": list(r["failure_tags"]), "primary_failure_type": r["primary_failure_type"], "stress_condition": r["stress_condition"], "locale": r["locale"]})
    st.json({"metadata": r["metadata"], "provenance": {"source_file": r["source_file"], "source_key": r["source_key"], "run_id": r["run_id"], "source_example_id": r["source_example_id"]}})


# --- 9. Cascade simulator ------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_matched(baseline_dir: str) -> pd.DataFrame | None:
    from pathlib import Path

    return load_matched(Path(baseline_dir))


def render_cascade_simulator(ui: SimpleNamespace) -> None:
    ui.section("P2·9", "Cascade Simulator", "Rules → Jev → reference LLM → oracle, simulated from recorded predictions. No live inference.")
    matched = _cached_matched(str(baseline_dir()))
    if matched is None:
        _missing(ui, "baseline", "src/phase2/freeze_baseline.py create")
        return
    ui.callout("SIMULATED from recorded Phase I predictions. The 'oracle' returns the ground-truth label: it is NOT a measured human. Rules coverage is 0% because the repository has no rules baseline. Costs are ESTIMATES (recorded token counts × pricing table), latency is a simulated sequential path.")
    workload = st.selectbox("Workload (never pooled)", workloads(matched), key="p2_cas_workload")
    c1, c2 = st.columns(2)
    jev_t = c1.slider("Jev confidence threshold (accept Jev if ≥)", 0.50, 0.99, 0.90, 0.01, key="p2_cas_jev")
    llm_on = c2.checkbox("LLM fallback enabled", value=True, key="p2_cas_llm")
    llm_t = c2.slider("LLM validation threshold (accept LLM if confidence ≥)", 0.0, 1.0, 0.5, 0.05, key="p2_cas_llm_t", disabled=not llm_on)
    oracle_on = c1.checkbox("Oracle fallback enabled", value=True, key="p2_cas_oracle")
    s = run_cascade_simulation(matched, workload, jev_t, llm_on, oracle_on, llm_threshold=llm_t if llm_on else None)
    cards = [("Handled by Jev", _pct(s["coverage_jev"]), f"{s['n_jev']} of {s['n_total']}"), ("Handled by reference LLM", _pct(s["coverage_llm"]), f"{s['n_llm']} of {s['n_total']}"), ("Handled by oracle (simulated)", _pct(s["coverage_oracle"]), f"{s['n_oracle']} of {s['n_total']}")]
    if s["n_unhandled"]:
        cards.append(("Unhandled", _pct(s["coverage_unhandled"]), f"{s['n_unhandled']} (oracle off)"))
    st.markdown(ui.kpi_html(cards), unsafe_allow_html=True)
    st.markdown(ui.kpi_html([("Simulated accuracy (incl. oracle)", _pct(s["overall_accuracy_including_oracle"]), "oracle correct by construction"), ("Automated accuracy (excl. oracle)", _pct(s["automated_accuracy_excluding_oracle"]), f"{s['n_jev'] + s['n_llm']} automated requests"),
                             ("Est. automated cost / 1,000 requests", "n/a" if s["automated_stage_cost_per_1000_requests_usd_estimated"] is None else f"${s['automated_stage_cost_per_1000_requests_usd_estimated']:.4f}", "ESTIMATED · excludes oracle"), ("Simulated latency p50 / p95", f"{_num(s['path_latency_p50_ms'], 0)} / {_num(s['path_latency_p95_ms'], 0)} ms", f"SIMULATED · n={s['path_latency_n']} automated")]), unsafe_allow_html=True)
    st.caption(f"Stage errors among handled requests: {s['stage_errors']}. Fallback reasons: {s['fallback_reasons']}.")


def phase2_renderers() -> list:
    """Renderers in the same order as PHASE2_TABS."""
    return [render_calibration, render_risk_coverage, render_stability, render_context_stress, render_ood, render_adversarial, render_multilingual_reliability, render_failure_museum, render_cascade_simulator]
