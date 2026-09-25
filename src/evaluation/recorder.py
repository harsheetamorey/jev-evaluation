"""Serialize and load PredictionResult rows to/from a single results.parquet file."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from models.experiment import ExperimentConfig
from models.prediction import PredictionResult

DEFAULT_RESULTS_PATH = Path("data/results/results.parquet")


def results_to_dataframe(results: Sequence[PredictionResult], config: ExperimentConfig) -> pd.DataFrame:
    """Flatten results to rows, tagging each with the run_id of the experiment that produced it."""
    rows = [{"run_id": config.run_id, **dataclasses.asdict(result)} for result in results]
    return pd.DataFrame(rows)


def append_results(
    results: Sequence[PredictionResult],
    config: ExperimentConfig,
    path: Path = DEFAULT_RESULTS_PATH,
) -> None:
    """Append `results` to the Parquet file at `path`, creating it if needed.

    Re-running the same ExperimentConfig REPLACES that run's rows rather than
    adding a second copy. `run_id` is derived deterministically from the config
    (see models.experiment), so identical config means "the same run" -- blindly
    appending would leave two copies under one run_id, and every downstream
    metric would silently average them while reporting a doubled n.

    Writes the combined table to a temp file and replaces `path` atomically, so a
    crash mid-write can't corrupt previously recorded results.
    """
    if not results:
        return
    new_rows = results_to_dataframe(results, config)
    if path.exists():
        existing = pd.read_parquet(path)
        superseded = int((existing["run_id"] == config.run_id).sum()) if "run_id" in existing.columns else 0
        if superseded:
            existing = existing[existing["run_id"] != config.run_id]
            print(f"note: replaced {superseded} existing row(s) for run_id {config.run_id}")
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_rows
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load all recorded results from `path`."""
    return pd.read_parquet(path)
