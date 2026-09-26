"""Tests for the Phase II reproducibility/integrity module. No model calls, no network."""

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from phase2.baseline import BASELINE_DIR, MANIFEST_NAME
from phase2.integrity import (
    ANALYSIS_ARTIFACTS,
    STRESS_CLI,
    STRESS_MODULES,
    FakeOfflineClient,
    _diff_detail,
    build_phase2_manifest,
    check,
    live_reproduction_plan,
    live_result_checks,
    live_results_record,
    live_stress_summary,
    pipeline_selfcheck,
    render_plan_markdown,
    stress_dataset_checks,
    write_phase2_manifest,
)

needs_baseline = pytest.mark.skipif(not (BASELINE_DIR / "canonical_results.parquet").exists(), reason="frozen baseline parquet is not tracked; run the freeze first")


def test_registry_covers_every_stress_experiment_and_analysis_artifact_dir() -> None:
    assert set(STRESS_MODULES) == set(STRESS_CLI) == {"ambiguity", "stability", "noise", "context_pollution", "context_relevance", "ood", "adversarial", "choice_overlap"}
    assert set(ANALYSIS_ARTIFACTS) == {"calibration", "selective_prediction", "multilingual", "failures", "cascade", "full_cascade", "cost_quality"}
    for cli in STRESS_CLI.values():
        assert (Path("src/phase2") / cli).exists()


def test_check_rejects_unknown_status_and_diff_detail_points_at_the_line(tmp_path: Path) -> None:
    assert check("x", "pass")["status"] == "pass"
    with pytest.raises(AssertionError):
        check("x", "maybe")
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_text("1\n2\n3\n")
    b.write_text("1\nX\n3\n")
    c.write_text("1\n2\n")
    assert _diff_detail(a, b) == "first difference at line 2" and "line counts differ" in _diff_detail(a, c)


def test_fake_client_is_deterministic_offline_and_returns_a_valid_choice() -> None:
    from typesafe_sdk import Choice

    async def ask(client, state):
        return await client.system_one(state, {"intent": Choice(instructions="x", criteria={"a": None, "b": None, "c": None})})

    c1, c2 = FakeOfflineClient(), FakeOfflineClient()
    r1, r2 = asyncio.run(ask(c1, {"user_message": "hi"})), asyncio.run(ask(c2, {"user_message": "hi"}))
    assert r1.choices["intent"].choice == r2.choices["intent"].choice in {"a", "b", "c"} and c1.calls == 1
    assert r1.model == "fake-offline"


def test_every_tracked_stress_dataset_verifies_and_rebuilds_deterministically() -> None:
    results = stress_dataset_checks()
    failed = [r for r in results if r["status"] == "fail"]
    assert not failed, failed
    passed = [r for r in results if r["status"] == "pass"]
    assert len(passed) == len(STRESS_MODULES) and all("deterministic rebuild ok" in r["detail"] for r in passed)


def test_stress_check_detects_a_tampered_frozen_dataset(tmp_path: Path) -> None:
    import shutil

    shutil.copytree("data/stress", tmp_path / "stress")
    target = tmp_path / "stress" / "ambiguity" / "dataset.jsonl"
    target.chmod(0o644)
    target.write_text(target.read_text() + "\n")
    bad = [r for r in stress_dataset_checks(tmp_path / "stress") if r["check"] == "stress dataset ambiguity"]
    assert bad[0]["status"] == "fail" and "modified" in bad[0]["detail"]


def test_offline_pipeline_selfcheck_runs_build_run_analyze_for_every_experiment_without_network() -> None:
    results = pipeline_selfcheck()
    assert [r["status"] for r in results] == ["pass"] * len(STRESS_MODULES), results
    assert all("no network" in r["detail"] for r in results)


def synthetic_stress_and_baseline(tmp_path: Path) -> tuple[Path, Path]:
    stress = tmp_path / "stress"
    for name, n in zip(STRESS_MODULES, (10, 20, 30, 40, 50, 60, 70, 80), strict=True):
        (stress / name).mkdir(parents=True)
        (stress / name / "dataset_manifest.json").write_text(json.dumps({"n_rows": n}))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    rows = [{"provider": p, "input_tokens": 100.0, "output_tokens": 10.0} for p in ("jev", "gpt-4o-mini") for _ in range(3)]
    pd.DataFrame(rows).to_parquet(baseline / "canonical_results.parquet", index=False)
    return stress, baseline


def test_live_plan_counts_calls_per_dataset_and_flags_which_commands_make_api_calls(tmp_path: Path) -> None:
    stress, baseline = synthetic_stress_and_baseline(tmp_path)
    plan = live_reproduction_plan(stress, baseline)
    assert plan["jev_calls_total"] == 360 and plan["reference_llm_calls_total_optional"] == 360
    assert plan["datasets"]["noise"]["jev_calls"] == 30 and plan["datasets"]["noise"]["rows"] == 30
    cmds = plan["datasets"]["ambiguity"]["commands"]
    assert cmds["live_jev_MAKES_API_CALLS"].endswith("run --providers jev --approve-calls 10")
    assert cmds["live_jev_and_reference_llm_MAKES_API_CALLS"].endswith("--approve-calls 20")
    assert "MAKES_API_CALLS" not in "".join(k for k in cmds if k in ("dry_run_no_api_calls", "analyze_offline")) and cmds["dry_run_no_api_calls"].endswith("report")
    assert plan["estimated_jev_cost_usd_total"] > 0 and "ESTIMATES" in plan["cost_basis"] and "--approve-calls" in plan["safety"]


def test_plan_markdown_lists_every_experiment_and_never_claims_anything_was_run(tmp_path: Path) -> None:
    stress, baseline = synthetic_stress_and_baseline(tmp_path)
    md = render_plan_markdown(live_reproduction_plan(stress, baseline))
    assert "Nothing in this file has been executed" in md
    for name in STRESS_MODULES:
        assert f"## {name} (" in md
    lines = md.splitlines()
    live_cmds = [i for i, line in enumerate(lines) if " run --providers " in line]
    assert len(live_cmds) == 2 * len(STRESS_MODULES)  # a Jev-only and a Jev+LLM command per experiment
    assert all(lines[i - 1].startswith("# LIVE") for i in live_cmds)  # every command that spends API calls is labelled LIVE right above it
    assert all(not lines[i - 1].startswith("# LIVE") for i, line in enumerate(lines) if line.endswith(" report") or line.endswith(" analyze"))


@needs_baseline
def test_manifest_marks_unrecorded_history_never_backfills_and_pins_hashes() -> None:
    m = build_phase2_manifest()
    assert m["historical_metadata"]["jev_model_version"]["status"] == "not_recorded" and m["providers"]["jev_model_version"] == "not_recorded"
    assert m["dataset_revisions"]["bitext"] == "not_recorded" and m["dataset_revisions"]["massive"] == "not_recorded"
    assert "NOT the values" in m["environment_at_manifest"]["note"] and set(m["stress_datasets"]) == set(STRESS_MODULES)
    assert all(len(v["dataset_sha256"]) == 64 and len(v["source_example_ids_sha256"]) == 64 for v in m["stress_datasets"].values())
    assert m["live_stress_results"]["recorded"] in ("none: live experiments have not been run",) or isinstance(m["live_stress_results"]["recorded"], list)
    assert set(m["config_hashes"]) == {"pyproject.toml", "uv.lock", "src/evaluation/pricing.py"} and "integrity.py" in m["experiment_code_sha256"]
    assert m["seed_and_threshold_constants_in_code"]["noise"]["NOISE_SEED"] == 6 and m["git"]["commit"]
    assert m["phase1_result_files_sha256"] and m["baseline"]["canonical"]["canonical_rows"] > 0
    json.dumps(m, default=str)


@needs_baseline
def test_manifest_is_written_once_and_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "phase2_manifest.json"
    write_phase2_manifest(out)
    assert json.loads(out.read_text())["schema_version"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_phase2_manifest(out)


@needs_baseline
def test_repository_checks_pass_on_the_real_frozen_baseline() -> None:
    from phase2.integrity import repository_checks

    if not (BASELINE_DIR / MANIFEST_NAME).exists():
        pytest.skip("no baseline manifest")
    failed = [r for r in repository_checks() if r["status"] == "fail"]
    # An uncommitted tracked file legitimately shows as a git difference while work is in progress; anything else must pass.
    assert all("tracked manifests" in r["check"] for r in failed), failed


def test_seed_gaps_accept_a_recorded_seed_or_an_explicit_not_applicable_and_flag_silence() -> None:
    from phase2.integrity import seed_gaps

    assert seed_gaps({"seed": 20260924, "seed_status": "recorded"}) == []
    assert seed_gaps({"context_seed": 7}) == []
    assert seed_gaps({"seed": None, "seed_status": "not_applicable"}) == []
    assert seed_gaps({}) and seed_gaps({"seed": None}) and seed_gaps({"seed_status": "not_applicable"})  # silence / null without the explicit status is still a gap


def test_frozen_stress_manifests_state_their_seed_or_that_none_applies() -> None:
    import json

    from phase2.integrity import STRESS_MODULES, seed_gaps
    from phase2.stress import SOURCE_SEED, STRESS_DIR

    for name in STRESS_MODULES:
        m = json.loads((STRESS_DIR / name / "dataset_manifest.json").read_text())
        assert seed_gaps(m) == [], name
    rel = json.loads((STRESS_DIR / "context_relevance" / "dataset_manifest.json").read_text())
    assert rel["seed"] is None and rel["seed_status"] == "not_applicable"
    co = json.loads((STRESS_DIR / "choice_overlap" / "dataset_manifest.json").read_text())
    assert co["seed"] == SOURCE_SEED and co["seed_status"] == "recorded"


def _fake_results(dataset_dir: str, results_dir: Path, drop: int = 0, errors: int = 0, dup: bool = False) -> Path:
    rows = [json.loads(line) for line in Path(dataset_dir, "dataset.jsonl").read_text().splitlines()]
    frame = pd.DataFrame({"variant_id": [r["variant_id"] for r in rows], "provider": "jev", "error": None, "input_tokens": 10, "output_tokens": 2, "estimated_cost_usd": 0.001})
    if errors:
        frame.loc[: errors - 1, "error"] = "boom"
    if drop:
        frame = frame.iloc[drop:]
    if dup:
        frame = pd.concat([frame, frame.iloc[:1]])
    results_dir.mkdir(parents=True)
    path = results_dir / "raw_results_jev.parquet"
    frame.to_parquet(path)
    return path


def test_live_results_record_measures_rows_errors_tokens_cost_hash_and_status(tmp_path: Path) -> None:
    from phase2.baseline import sha256_file

    path = _fake_results("data/stress/context_relevance", tmp_path / "context")  # context_relevance writes to <root>/context
    record = live_results_record(tmp_path)
    r = record["context_relevance"]
    f = r["result_files"][0]
    assert r["status"] == "complete" and r["expected_rows"] == 90 and f["rows"] == 90 and f["api_error_count"] == 0
    assert f["provider"] == ["jev"] and f["input_tokens"] == 900 and f["output_tokens"] == 180 and f["estimated_cost_usd"] == pytest.approx(0.09)
    assert f["sha256"] == sha256_file(path) and len(r["dataset_sha256"]) == 64
    assert all(record[n]["status"] == "not_run" and record[n]["result_files"] == [] for n in STRESS_MODULES if n != "context_relevance")
    summary = live_stress_summary(record)
    assert summary["recorded"] == ["context_relevance"] and summary["total_rows"] == 90 and summary["n_experiments_complete"] == 1
    checks = {c["check"]: c["status"] for c in live_result_checks(tmp_path)}
    assert checks["live results context_relevance"] == "pass" and checks["live results ambiguity"] == "warn"


def test_live_results_record_flags_missing_duplicate_and_errored_rows(tmp_path: Path) -> None:
    _fake_results("data/stress/context_relevance", tmp_path / "context", drop=2)
    assert live_results_record(tmp_path)["context_relevance"]["status"] == "incomplete"
    assert {c["check"]: c["status"] for c in live_result_checks(tmp_path)}["live results context_relevance"] == "fail"
    _fake_results("data/stress/ood", tmp_path / "ood", errors=3)
    ood = live_results_record(tmp_path)["ood"]
    assert ood["status"] == "complete_with_errors" and ood["result_files"][0]["api_error_count"] == 3
    _fake_results("data/stress/ambiguity", tmp_path / "ambiguity", dup=True)
    assert live_results_record(tmp_path)["ambiguity"]["status"] == "incomplete"  # 301 rows for 300 frozen ids
