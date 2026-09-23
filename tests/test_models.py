"""Unit tests for the shared PredictionResult / Example / ExperimentConfig models."""

import dataclasses

import pytest

from models.experiment import ExperimentConfig
from models.prediction import Example, PredictionResult


def test_prediction_result_holds_given_fields() -> None:
    result = PredictionResult(
        experiment="bitext_hard_choice",
        provider="jev",
        example_id="ex-1",
        dataset="bitext",
        locale="en",
        ground_truth="track_order",
        prediction="track_order",
        correct=True,
        confidence=0.92,
        latency_ms=123.4,
        candidates=["track_order", "cancel_order"],
        error=None,
    )
    assert result.correct is True
    assert result.candidates == ["track_order", "cancel_order"]


def test_example_defaults_ground_truth_and_locale_to_none() -> None:
    example = Example(
        example_id="ex-1",
        dataset="bitext",
        state="Where is my order?",
        candidates=["track_order", "other"],
    )
    assert example.ground_truth is None
    assert example.locale is None


def test_experiment_config_run_id_is_deterministic() -> None:
    config_a = ExperimentConfig(name="bitext_hard_choice", dataset="bitext", providers=("jev",), seed=0)
    config_b = ExperimentConfig(name="bitext_hard_choice", dataset="bitext", providers=("jev",), seed=0)
    assert config_a.run_id == config_b.run_id


def test_experiment_config_run_id_ignores_provider_order() -> None:
    config_a = ExperimentConfig(name="x", dataset="bitext", providers=("jev", "llm"), seed=0)
    config_b = ExperimentConfig(name="x", dataset="bitext", providers=("llm", "jev"), seed=0)
    assert config_a.run_id == config_b.run_id


@pytest.mark.parametrize(
    "other_kwargs",
    [
        {"seed": 1},
        {"name": "other_experiment"},
        {"dataset": "massive"},
        {"providers": ("llm",)},
    ],
)
def test_experiment_config_run_id_changes_with_any_field(other_kwargs: dict) -> None:
    base = {"name": "x", "dataset": "bitext", "providers": ("jev",), "seed": 0}
    config_a = ExperimentConfig(**base)
    config_b = ExperimentConfig(**{**base, **other_kwargs})
    assert config_a.run_id != config_b.run_id


def test_experiment_config_is_frozen() -> None:
    config = ExperimentConfig(name="x", dataset="bitext", providers=("jev",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]
