"""CPU-only contracts for the dummy task, models, splits and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from combinatorial_search_dynamics.data.splits import make_instance_split, read_split, write_split
from combinatorial_search_dynamics.dynamics.transition import MarkovTransitionModel
from combinatorial_search_dynamics.evaluation.metrics import evaluate_trajectories
from combinatorial_search_dynamics.features.external import extract_external_features
from combinatorial_search_dynamics.state_models.baseline import HammingDistanceStateModel
from combinatorial_search_dynamics.tasks.dummy_binary import DummyBinaryTask
from combinatorial_search_dynamics.tasks.registry import get_task


def test_dummy_task_contract_and_seed() -> None:
    task = DummyBinaryTask(bit_count=5, max_steps=3)
    instance = task.generate_instance(32)
    assert instance == task.generate_instance(32)
    assert instance != task.generate_instance(33)
    initial = task.initial_state(instance)
    assert initial.bits == instance.initial_bits
    assert task.legal_actions(initial) == tuple(range(5))
    assert task.objective(initial) == sum(a != b for a, b in zip(initial.bits, initial.target_bits))
    following = task.apply_action(initial, 1)
    assert initial.step == 0 and following.step == 1
    assert initial.bits != following.bits
    assert task.deserialize_state(task.serialize_state(following)) == following
    assert task.parse_action("1") == 1
    with pytest.raises(ValueError, match="action"):
        task.apply_action(initial, 5)
    with pytest.raises(ValueError, match="action"):
        task.parse_action("invalid")
    with pytest.raises(KeyError, match="Unknown task"):
        get_task("unknown", bit_count=5, max_steps=3)
    assert task.is_terminal(task.initial_state(task.generate_instance(32))) is False
    with pytest.raises(ValueError, match="bit_count"):
        DummyBinaryTask(bit_count=0, max_steps=3)


def test_instance_split_reproducibility_and_test_fit_guard(tmp_path: Path) -> None:
    ids = [f"ins_{index}" for index in range(12)]
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    split = make_instance_split(ids, seed=33, ratios=ratios)
    assert split == make_instance_split(ids[::-1], seed=33, ratios=ratios)
    assert set(split.assignments) == set(ids)
    assert all(split.instance_ids(name) for name in ratios)
    assert set(split.fit_instance_ids()).isdisjoint(split.instance_ids("test"))
    with pytest.raises(ValueError, match="train"):
        split.fit_instance_ids("test")
    path = tmp_path / "splits.json"
    write_split(path, split)
    assert read_split(path) == split
    with pytest.raises(FileExistsError):
        write_split(path, split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignments"][ids[0]] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_split(path)
    with pytest.raises(ValueError, match="sum"):
        make_instance_split(ids, seed=1, ratios={"train": 0.8, "validation": 0.2, "test": 0.2})
    with pytest.raises(ValueError, match="At least three"):
        make_instance_split(ids[:2], seed=1, ratios=ratios)


def test_external_features_and_state_model_never_need_trial_final_outcome(tmp_path: Path) -> None:
    point = {
        "instance_id": "ins_train",
        "trial_id": "trl_train",
        "checkpoint_id": "chk_train",
        "checkpoint_index": 0,
        "state": {"bits": [0, 1], "target_bits": [1, 1]},
    }
    features = extract_external_features([point])
    assert features[0]["hamming_distance"] == 1
    model = HammingDistanceStateModel(bit_count=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(features)
    with pytest.raises(ValueError, match="train"):
        model.fit(features, fit_split="test", split_hash="x", instance_ids=["ins_train"])
    assert (
        model.fit_transform(
            features, fit_split="train", split_hash="split123", instance_ids=["ins_train"]
        )[0]["discrete_state"]
        == 1
    )
    test_data = [{**features[0], "instance_id": "ins_test", "hamming_distance": 0}]
    assert model.transform(test_data)[0]["discrete_state"] == 0
    path = tmp_path / "state.json"
    model.save(path)
    restored = HammingDistanceStateModel.load(path)
    assert restored.transform(test_data) == model.transform(test_data)
    assert restored.fit_metadata["fit_split"] == "train"
    assert restored.fit_metadata["instance_ids"] == ["ins_train"]
    with pytest.raises(ValueError, match="outside"):
        HammingDistanceStateModel(2).fit(
            test_data, fit_split="train", split_hash="x", instance_ids=["ins_train"]
        )


def test_markov_rows_absorbing_horizons_and_saved_metadata(tmp_path: Path) -> None:
    model = MarkovTransitionModel(state_count=3, smoothing=0)
    with pytest.raises(ValueError, match="train"):
        model.fit([[2, 1]], fit_split="test", split_hash="x", instance_ids=["i"])
    model.fit([[2, 1, 0], [2, 1, 1]], fit_split="train", split_hash="abc", instance_ids=["i"])
    matrix = model.transition_matrix
    assert np.all(matrix >= 0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_array_equal(matrix[0], [1, 0, 0])
    np.testing.assert_array_equal(model.predict_distribution(2, 1), matrix[2])
    np.testing.assert_allclose(
        model.predict_distribution(2, 2), np.linalg.matrix_power(matrix, 2)[2]
    )
    assert model.score([[2, 1, 0]])["n_transitions"] == 2
    path = tmp_path / "dynamics.json"
    model.save(path)
    restored = MarkovTransitionModel.load(path)
    np.testing.assert_allclose(
        restored.predict_distribution(2, 2), model.predict_distribution(2, 2)
    )
    assert restored.fit_metadata["split_hash"] == "abc"
    with pytest.raises(ValueError, match="state"):
        model.predict_distribution(3, 1)
    with pytest.raises(ValueError, match="horizon"):
        model.predict_distribution(1, -1)
    unseen = MarkovTransitionModel(4, 0).fit(
        [[2, 1]], fit_split="train", split_hash="x", instance_ids=["i"]
    )
    np.testing.assert_array_equal(unseen.transition_matrix[3], [0, 0, 0, 1])


def test_known_heldout_metrics_and_probability_clipping() -> None:
    model = MarkovTransitionModel(3, smoothing=0).fit(
        [[2, 1, 0], [2, 1, 1]], fit_split="train", split_hash="x", instance_ids=["train"]
    )
    test = [
        {
            "instance_id": "test",
            "trial_id": "trial",
            "states": [2, 1, 0],
            "checkpoint_ids": ["a", "b", "c"],
        }
    ]
    result = evaluate_trajectories(model, test)
    assert result.metrics["one_step_nll"] == pytest.approx(np.log(2) / 2)
    assert result.metrics["one_step_accuracy"] == 1
    assert result.metrics["success_brier_score"] == pytest.approx(0.125)
    assert (result.n_transitions, result.n_instances, result.n_trials) == (2, 1, 1)
    assert result.per_instance[0]["instance_id"] == "test"
    unseen = [{**test[0], "states": [1, 2], "checkpoint_ids": ["a", "b"]}]
    clipped = evaluate_trajectories(model, unseen, probability_floor=1e-6)
    assert clipped.metrics["one_step_nll"] == pytest.approx(-np.log(1e-6))
