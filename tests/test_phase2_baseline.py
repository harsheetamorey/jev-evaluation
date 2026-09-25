"""Tests for Phase II baseline manifest creation, loading and verification."""

import json
from pathlib import Path

import pytest

import evaluation.runner as runner_module
from evaluation.recorder import append_results
from models.experiment import ExperimentConfig
from models.prediction import PredictionResult
import pandas as pd

from phase2.baseline import (
    CANONICAL_NAME,
    MANIFEST_NAME,
    NOT_RECORDED,
    build_canonical,
    dedup_summary,
    QUESTION_CONFIG,
    BaselineError,
    create_baseline,
    load_manifest,
    sha256_file,
    verify_baseline,
)
from phase2.freeze_baseline import main


def _result(example_id: str, provider: str) -> PredictionResult:
    return PredictionResult(
        experiment="bitext_hard_choice",
        provider=provider,
        example_id=example_id,
        dataset="bitext",
        locale=None,
        ground_truth="a",
        prediction="a",
        correct=True,
        confidence=0.9,
        latency_ms=10.0,
        candidates=["a", "b"],
        error=None,
    )


@pytest.fixture
def phase1(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results"
    samples_dir = tmp_path / "samples"
    results_dir.mkdir()
    samples_dir.mkdir()
    (samples_dir / "bitext_dev.jsonl").write_text('{"id": "x-0"}\n')
    ids = [f"bitext-test-{i}" for i in range(50)]
    config = ExperimentConfig(name="bitext_hard_choice", dataset="bitext:dev", providers=("jev", "gpt-4o-mini"), seed=0)
    append_results([_result(i, p) for p in ("jev", "gpt-4o-mini") for i in ids], config, path=results_dir / "results.parquet")
    (results_dir / "results_metrics.json").write_text("{}")
    return results_dir, samples_dir


def _create(phase1: tuple[Path, Path], tmp_path: Path) -> Path:
    results_dir, samples_dir = phase1
    out = tmp_path / "baseline"
    create_baseline(results_dir=results_dir, samples_dir=samples_dir, out_dir=out)
    return out


def test_create_writes_manifest_with_required_fields(phase1, tmp_path: Path) -> None:
    out = _create(phase1, tmp_path)
    manifest = load_manifest(out / MANIFEST_NAME)
    for key in ("created_at", "historical_metadata", "environment_at_freeze", "canonical", "runs", "samples", "derived_files"):
        assert key in manifest
    run = manifest["runs"][0]
    assert run["n_unique_examples"] == 50 and run["example_ids"][0] == "bitext-test-0"
    assert run["config"]["dataset"] == "bitext:dev"  # recovered via run_id hash
    assert set(run["raw_metrics_by_provider"]) == {"jev", "gpt-4o-mini"}
    assert (run["raw_rows"], run["canonical_rows"], run["duplicated_identities"]) == (100, 100, 0)


def test_create_never_modifies_phase1_files(phase1, tmp_path: Path) -> None:
    results_dir, _ = phase1
    before = {p.name: sha256_file(p) for p in results_dir.iterdir()}
    _create(phase1, tmp_path)
    assert {p.name: sha256_file(p) for p in results_dir.iterdir()} == before


def test_create_refuses_to_overwrite_existing_baseline(phase1, tmp_path: Path) -> None:
    out = _create(phase1, tmp_path)
    with pytest.raises(BaselineError, match="refusing to overwrite"):
        create_baseline(results_dir=phase1[0], samples_dir=phase1[1], out_dir=out)


def test_verify_passes_then_detects_tampering_and_source_drift(phase1, tmp_path: Path) -> None:
    out = _create(phase1, tmp_path)
    assert verify_baseline(out / MANIFEST_NAME, samples_dir=phase1[1]) == ([], [])

    (phase1[0] / "results_metrics.json").write_text('{"changed": true}')
    errors, warnings = verify_baseline(out / MANIFEST_NAME, samples_dir=phase1[1])
    assert errors == [] and any("source changed" in w for w in warnings)

    frozen = out / "results_metrics.json"
    frozen.chmod(0o644)
    frozen.write_text("tampered")
    errors, _ = verify_baseline(out / MANIFEST_NAME, samples_dir=phase1[1])
    assert errors == ["frozen file modified: results_metrics.json"]


def test_load_manifest_rejects_missing_and_unknown_schema(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="No baseline manifest"):
        load_manifest(tmp_path / MANIFEST_NAME)
    bad = tmp_path / MANIFEST_NAME
    bad.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(BaselineError, match="schema_version"):
        load_manifest(bad)


def test_cli_create_then_verify(phase1, tmp_path: Path) -> None:
    out = tmp_path / "cli_baseline"
    assert main(["create", "--results-dir", str(phase1[0]), "--out", str(out)]) == 0
    assert main(["verify", "--out", str(out)]) == 0
    assert main(["create", "--results-dir", str(phase1[0]), "--out", str(out)]) == 1


def test_question_config_matches_runner_source() -> None:
    assert QUESTION_CONFIG["instructions"] in Path(runner_module.__file__).read_text()


def test_missing_historical_metadata_is_marked_not_recorded(phase1, tmp_path: Path) -> None:
    hist = load_manifest(_create(phase1, tmp_path) / MANIFEST_NAME)["historical_metadata"]
    assert hist["jev_model_version"] == {"value": None, "status": NOT_RECORDED}
    assert hist["sdk_version"]["status"] == NOT_RECORDED
    assert hist["question_config"] == {"value": None, "status": NOT_RECORDED}
    assert hist["concurrency"] == {"value": None, "status": NOT_RECORDED, "current_cli_default": 5}
    for dataset in hist["dataset_revision"].values():
        assert dataset["value"] is None and dataset["status"] == NOT_RECORDED
    assert hist["sample_seeds"]["status"] == NOT_RECORDED


def _raw_with_repeats() -> pd.DataFrame:
    def row(example_id: str, prediction: str, request_id: str, cost: float | None) -> dict:
        r = _result(example_id, "jev")
        return {"run_id": "r1", **{**r.__dict__, "prediction": prediction, "request_id": request_id, "estimated_cost_usd": cost}}

    return pd.DataFrame(
        [
            row("a", "x", "req1", None),  # 0: repeat of 3, same prediction
            row("b", "x", "req2", None),  # 1: conflicts with 4
            row("c", "x", "req3", None),  # 2: unique
            row("a", "x", "req4", 0.001),  # 3
            row("b", "y", "req5", 0.001),  # 4
        ]
    )


def test_build_canonical_keeps_latest_row_regardless_of_agreement() -> None:
    canonical, report = build_canonical(_raw_with_repeats())
    assert canonical["raw_row"].tolist() == [2, 3, 4]  # c unique, later a, later b
    assert set(canonical["example_id"]) == {"a", "b", "c"}  # every logical example survives, conflicting or not
    by_id = report.set_index("example_id")
    assert by_id.loc["a", "predictions_agree"] and not by_id.loc["b", "predictions_agree"]
    assert (by_id.loc["b", "earlier_prediction"], by_id.loc["b", "later_prediction"]) == ("x", "y")
    assert by_id.loc["b", "selected_raw_row"] == 4 and by_id.loc["b", "earlier_request_id"] == "req2"
    summary = dedup_summary(5, len(canonical), report)
    assert (summary["duplicated_identities"], summary["agreeing_repeated_calls"], summary["disagreeing_repeated_calls"]) == (2, 1, 1)
    assert summary["repeat_prediction_agreement"] == 0.5 and summary["rows_removed"] == 2


def test_manifest_records_raw_canonical_and_duplicate_counts(phase1, tmp_path: Path) -> None:
    results_dir, samples_dir = phase1
    config = ExperimentConfig(name="bitext_hard_choice", dataset="bitext:small", providers=("jev",), seed=0)
    path = results_dir / "results.parquet"
    append_results([_result("dup", "jev")], config, path=path)
    raw = pd.read_parquet(path)
    pd.concat([raw, raw[raw["example_id"] == "dup"]], ignore_index=True).to_parquet(path, index=False)  # simulate a repeated row

    manifest = load_manifest(_create((results_dir, samples_dir), tmp_path) / MANIFEST_NAME)
    canon = manifest["canonical"]
    assert canon["raw_rows"] - canon["canonical_rows"] == canon["rows_removed"] == 1
    assert canon["duplicated_identities"] == 1 and canon["repeat_prediction_agreement"] == 1.0 and "LATEST" in canon["rule"]
    assert manifest["derived_files"][CANONICAL_NAME]["rows"] == canon["canonical_rows"]


def test_verify_detects_canonical_tampering(phase1, tmp_path: Path) -> None:
    out = _create(phase1, tmp_path)
    canonical = out / CANONICAL_NAME
    canonical.chmod(0o644)
    pd.read_parquet(canonical).iloc[:-1].to_parquet(canonical, index=False)
    errors, _ = verify_baseline(out / MANIFEST_NAME, samples_dir=phase1[1])
    assert f"derived file modified: {CANONICAL_NAME}" in errors
