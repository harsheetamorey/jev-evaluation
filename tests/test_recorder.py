"""Unit tests for Parquet serialization/loading and append-safe writes."""

from pathlib import Path

from evaluation.recorder import append_results, load_results
from models.experiment import ExperimentConfig
from models.prediction import PredictionResult


def _result(example_id: str) -> PredictionResult:
    return PredictionResult(
        experiment="unit-test",
        provider="fake",
        example_id=example_id,
        dataset="bitext",
        locale="en",
        ground_truth="x",
        prediction="x",
        correct=True,
        confidence=0.9,
        latency_ms=12.0,
        candidates=["x", "y"],
        error=None,
    )


def test_append_results_creates_file_with_run_id_column(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    config = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))

    append_results([_result("ex-1")], config, path=path)

    df = load_results(path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == config.run_id
    assert df.iloc[0]["example_id"] == "ex-1"


def test_append_results_accumulates_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    config = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))

    append_results([_result("ex-1")], config, path=path)
    append_results([_result("ex-2"), _result("ex-3")], config, path=path)

    df = load_results(path)
    assert sorted(df["example_id"]) == ["ex-1", "ex-2", "ex-3"]


def test_append_results_with_empty_list_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    config = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))

    append_results([], config, path=path)

    assert not path.exists()


def test_candidates_list_column_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    config = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))

    append_results([_result("ex-1")], config, path=path)

    df = load_results(path)
    assert list(df.iloc[0]["candidates"]) == ["x", "y"]
