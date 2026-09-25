"""Tests for the shared stress-experiment infrastructure. No network, no API key."""

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from typesafe_sdk import TypeSafeError

from models.prediction import Example
from phase2.stress import (
    StressError,
    StressEvaluator,
    execute_live,
    grouped_paired_metrics,
    load_frozen_dataset,
    pair_results,
    paired_metrics,
    plan_calls,
    select_bitext_sources,
    seeded_rng,
    stable_int,
    top_margin,
    validate_rows,
    write_frozen_dataset,
)


def row(vid: str, src: str = "s1", **extra) -> dict:
    return {"variant_id": vid, "source_example_id": src, "text": "hello", "candidates": ["a", "b"], "ground_truth": "a", **extra}


class _Answer:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choice, self.probabilities = choice, probabilities


class _Usage:
    input_tokens, output_tokens = 10, 2


class _Response:
    def __init__(self, choice: str, probabilities: dict[str, float]) -> None:
        self.choices = {"intent": _Answer(choice, probabilities)}
        self.model, self.usage, self.request_id = "fake", _Usage(), "req-1"


class _Client:
    def __init__(self, choice: str = "a", error: Exception | None = None) -> None:
        self.calls, self._choice, self._error = [], choice, error

    async def system_one(self, state, questions):
        self.calls.append(state)
        if self._error:
            raise self._error
        return _Response(self._choice, {"a": 0.7, "b": 0.3})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def test_stable_hash_is_deterministic_and_seeded_rng_repeats() -> None:
    assert stable_int("x", 1) == stable_int("x", 1) != stable_int("x", 2)
    assert seeded_rng("a", 1).random() == seeded_rng("a", 1).random()


def test_select_bitext_sources_is_deterministic_and_unique() -> None:
    a, b = select_bitext_sources(20), select_bitext_sources(20)
    assert [r.id for r in a] == [r.id for r in b] and len({r.id for r in a}) == 20


def test_validate_rows_requires_contract_fields_and_unique_ids() -> None:
    validate_rows([row("v1"), row("v2")])
    with pytest.raises(StressError, match="missing"):
        validate_rows([{"variant_id": "v1", "text": "t", "candidates": ["a"]}])  # no source_example_id
    with pytest.raises(StressError, match="duplicate"):
        validate_rows([row("v1"), row("v1")])
    with pytest.raises(StressError, match="no candidates"):
        validate_rows([{**row("v1"), "candidates": []}])


def test_frozen_dataset_roundtrip_is_deterministic_read_only_and_tamper_evident(tmp_path: Path) -> None:
    rows = [row("v2", "s2"), row("v1", "s1")]
    m1 = write_frozen_dataset(tmp_path / "a", rows, {"seed": 7})
    m2 = write_frozen_dataset(tmp_path / "b", rows, {"seed": 7})
    assert m1["sha256"] == m2["sha256"] and m1["n_source_examples"] == 2  # same input -> identical bytes
    loaded, manifest = load_frozen_dataset(tmp_path / "a")
    assert loaded == rows and manifest["seed"] == 7
    with pytest.raises(StressError, match="refusing to overwrite"):
        write_frozen_dataset(tmp_path / "a", rows, {})
    data = tmp_path / "a" / "dataset.jsonl"
    data.chmod(0o644)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(StressError, match="modified"):
        load_frozen_dataset(tmp_path / "a")
    with pytest.raises(StressError, match="No frozen dataset"):
        load_frozen_dataset(tmp_path / "missing")


def test_evaluator_records_full_probabilities_and_supports_custom_state() -> None:
    client = _Client("a")
    ev = StressEvaluator(client, "exp", "jev", state_builder=lambda e: {"message": e.state, "history": ["x"]})
    ex = Example(example_id="v1", dataset="d", state="hi", candidates=["a", "b"], ground_truth="a")
    result = asyncio.run(ev.predict(ex))
    assert (result.prediction, result.correct, result.confidence) == ("a", True, 0.7)
    assert ev.probabilities["v1"] == {"a": 0.7, "b": 0.3}
    assert client.calls == [{"message": "hi", "history": ["x"]}]


def test_evaluator_leaves_correct_null_without_ground_truth_and_records_errors() -> None:
    ev = StressEvaluator(_Client("a"), "exp", "jev")
    no_gt = Example(example_id="v1", dataset="d", state="hi", candidates=["a", "b"], ground_truth=None)
    assert asyncio.run(ev.predict(no_gt)).correct is None
    bad = StressEvaluator(_Client(error=TypeSafeError("boom")), "exp", "jev")
    err = asyncio.run(bad.predict(Example(example_id="v2", dataset="d", state="hi", candidates=["a"], ground_truth="a")))
    assert err.error and err.prediction is None and "v2" not in bad.probabilities


def test_top_margin_only_when_a_distribution_exists() -> None:
    assert top_margin({"a": 0.7, "b": 0.2, "c": 0.1}) == pytest.approx(0.5)
    assert top_margin({"a": 0.9}) is None and top_margin(None) is None and top_margin({}) is None


def test_plan_calls_counts_rows_times_providers() -> None:
    plan = plan_calls("exp", [row("v1"), row("v2"), row("v3")], ["jev", "openai:gpt-4o-mini"])
    assert (plan.n_rows, plan.calls_per_provider, plan.total_calls) == (3, 3, 6)


def test_live_run_requires_exact_call_approval_and_makes_no_client_before_it(tmp_path: Path) -> None:
    built: list[str] = []

    def factory(spec: str):
        built.append(spec)
        return _Client("a"), spec

    rows = [row("v1"), row("v2")]
    for bad in (None, 1, 3):
        with pytest.raises(StressError, match="approval"):
            asyncio.run(execute_live("exp", rows, "stress:exp", ["jev"], tmp_path, bad, client_factory=factory))
    assert built == []  # nothing constructed, therefore nothing called

    paths = asyncio.run(execute_live("exp", rows, "stress:exp", ["jev"], tmp_path, 2, client_factory=factory))
    df = pd.read_parquet(paths[0])
    assert len(df) == 2 and set(df["source_example_id"]) == {"s1"} and df["probabilities_json"].notna().all()
    with pytest.raises(StressError, match="refusing to overwrite"):  # recorded results are never clobbered
        asyncio.run(execute_live("exp", rows, "stress:exp", ["jev"], tmp_path, 2, client_factory=factory))


def _side(preds, correct, conf, src=None, provider="jev") -> pd.DataFrame:
    n = len(preds)
    return pd.DataFrame({"provider": provider, "source_example_id": src or [f"s{i}" for i in range(n)], "prediction": preds, "correct": correct, "confidence": conf, "latency_ms": 1.0})


def test_pairing_and_paired_metrics() -> None:
    base = _side(["a", "a", "b", "a"], [True, True, False, True], [0.9, 0.8, 0.6, 0.9])
    var = _side(["a", "b", "b", None], [True, False, False, None], [0.7, 0.5, 0.6, None])
    m = paired_metrics(pair_results(base, var))
    assert m["n_pairs"] == 3 and m["n_excluded_errors"] == 1  # errored variant excluded, not counted as a flip
    assert m["decision_flip_rate"] == pytest.approx(1 / 3) and m["label_agreement"] == pytest.approx(2 / 3)
    assert m["accuracy_base"] == pytest.approx(2 / 3) and m["accuracy_variant"] == pytest.approx(1 / 3)
    assert m["accuracy_delta"] == pytest.approx(-1 / 3)
    assert m["mean_confidence_delta"] == pytest.approx(((0.7 - 0.9) + (0.5 - 0.8) + (0.6 - 0.6)) / 3)
    assert m["mean_abs_confidence_delta"] == pytest.approx((0.2 + 0.3 + 0.0) / 3)


def test_pairing_rejects_ambiguous_base_and_compares_each_variant_to_its_own_source() -> None:
    dup = pd.concat([_side(["a"], [True], [0.9], ["s0"]), _side(["b"], [False], [0.5], ["s0"])])
    with pytest.raises(StressError, match="ambiguous"):
        pair_results(dup, _side(["a"], [True], [0.9], ["s0"]))
    base = _side(["a", "b"], [True, True], [0.9, 0.9], ["s0", "s1"])
    var = _side(["a", "a"], [True, False], [0.9, 0.4], ["s0", "s1"])
    paired = pair_results(base, var)
    assert set(paired["source_example_id"]) == {"s0", "s1"}
    assert paired.loc[paired["source_example_id"] == "s1", "prediction_base"].item() == "b"


def test_grouped_paired_metrics_and_empty_pairs() -> None:
    paired = pair_results(_side(["a", "a"], [True, True], [0.9, 0.9]), _side(["a", "b"], [True, False], [0.9, 0.5]))
    paired["kind"] = ["x", "y"]
    g = grouped_paired_metrics(paired, ["kind"]).set_index("kind")
    assert g.loc["x", "decision_flip_rate"] == 0.0 and g.loc["y", "decision_flip_rate"] == 1.0
    assert paired_metrics(paired.iloc[0:0])["n_pairs"] == 0


def test_live_run_sends_a_rows_state_payload_when_present(tmp_path: Path) -> None:
    client = _Client("a")
    rows = [row("v1"), {**row("v2"), "state_payload": {"message": "hello", "history": ["x", "y"]}}]
    asyncio.run(execute_live("exp", rows, "stress:exp", ["jev"], tmp_path, 2, client_factory=lambda spec: (client, spec)))
    assert {"user_message": "hello"} in client.calls and {"message": "hello", "history": ["x", "y"]} in client.calls
