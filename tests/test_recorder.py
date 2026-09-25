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


def test_append_results_accumulates_across_different_runs(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    config_a = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))
    config_b = ExperimentConfig(name="exp", dataset="bitext", providers=("other",))

    append_results([_result("ex-1")], config_a, path=path)
    append_results([_result("ex-2"), _result("ex-3")], config_b, path=path)

    df = load_results(path)
    assert sorted(df["example_id"]) == ["ex-1", "ex-2", "ex-3"]
    assert df["run_id"].nunique() == 2


def test_rerunning_the_same_config_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """run_id is derived from the config, so the same config is the same run.

    Appending a second copy would leave two sets of rows sharing one run_id,
    and every downstream metric would average them while reporting a doubled n.
    """
    path = tmp_path / "results.parquet"
    config = ExperimentConfig(name="exp", dataset="bitext", providers=("fake",))

    append_results([_result("ex-1"), _result("ex-2")], config, path=path)
    append_results([_result("ex-1"), _result("ex-2")], config, path=path)

    df = load_results(path)
    assert len(df) == 2
    assert sorted(df["example_id"]) == ["ex-1", "ex-2"]


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
