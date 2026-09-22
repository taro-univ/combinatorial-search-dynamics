"""Select only implemented Phase 2/3 state representations."""

from __future__ import annotations

from typing import Any

from combinatorial_search_dynamics.state_models.baseline import HammingDistanceStateModel
from combinatorial_search_dynamics.state_models.objective_gap import ObjectiveGapStateModel


def get_state_model(config: dict[str, Any], task_config: dict[str, Any]):
    name = config["name"]
    if name == "hamming_distance":
        if task_config["name"] != "dummy_binary":
            raise ValueError("Hamming distance state model requires dummy_binary")
        return HammingDistanceStateModel(bit_count=int(task_config["bit_count"]))
    if name == "objective_gap":
        if task_config["name"] != "knapsack":
            raise ValueError("Objective gap state model requires knapsack")
        return ObjectiveGapStateModel(bin_boundaries=config["bin_boundaries"])
    raise KeyError(f"Unknown state model: {name}")


def load_state_model(path):
    import json
    from pathlib import Path

    model_name = json.loads(Path(path).read_text(encoding="utf-8")).get("model")
    if model_name == "hamming_distance":
        return HammingDistanceStateModel.load(path)
    if model_name == "objective_gap":
        return ObjectiveGapStateModel.load(path)
    raise ValueError(f"Unsupported state model artifact: {model_name!r}")
