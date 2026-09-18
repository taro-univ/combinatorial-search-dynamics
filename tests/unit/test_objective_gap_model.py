"""Train-only discretization of proven-reference objective gaps."""

from __future__ import annotations

import pytest

from llm_search_dynamics.state_models.objective_gap import ObjectiveGapStateModel
from llm_search_dynamics.state_models.registry import get_state_model, load_state_model


def _rows(*gaps: float | None, identifier: str = "ins_train") -> list[dict]:
    return [{"instance_id": identifier, "optimality_gap": gap} for gap in gaps]


def test_gap_bins_train_metadata_and_save_load(tmp_path) -> None:
    model = ObjectiveGapStateModel(bin_boundaries=[0.1, 0.5])
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(_rows(0.0))
    with pytest.raises(ValueError, match="train"):
        model.fit(_rows(0.1), fit_split="test", split_hash="h", instance_ids=["ins_train"])
    model.fit(
        _rows(0.0, 0.05),
        fit_split="train",
        split_hash="h",
        instance_ids=["ins_train", "ins_unproven"],
    )
    assert model.fit_metadata["fit_split"] == "train"
    assert model.fit_metadata["instance_ids"] == ["ins_train"]
    assert model.fit_metadata["excluded_train_instance_ids"] == ["ins_unproven"]
    test = _rows(0.0, 0.05, 0.4, 0.9, identifier="ins_test")
    assert [row["discrete_state"] for row in model.transform(test)] == [0, 1, 2, 3]
    assert list(model.bin_boundaries) == [0.1, 0.5]  # test values cannot move bins
    path = tmp_path / "state_model.json"
    model.save(path)
    assert load_state_model(path).transform(test) == model.transform(test)
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError):
        model.transform(_rows(None))
    with pytest.raises(ValueError):
        model.fit(
            _rows(0.1, identifier="ins_test"),
            fit_split="train",
            split_hash="h",
            instance_ids=["ins_train"],
        )
    with pytest.raises(KeyError):
        get_state_model({"name": "unknown"}, {"name": "knapsack"})
