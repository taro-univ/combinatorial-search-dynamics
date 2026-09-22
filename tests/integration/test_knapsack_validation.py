"""Phase 3 structured validation rejects corrupted checkpoint facts."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from combinatorial_search_dynamics.config import resolved_config
from combinatorial_search_dynamics.data.parquet import read_parquet
from combinatorial_search_dynamics.data.reference_validation import validate_knapsack_dataset
from combinatorial_search_dynamics.data.schemas import get_schema
from combinatorial_search_dynamics.pipeline.stages import PilotPaths, reproduce_pilot
from tests.integration.test_knapsack_pipeline import knapsack_overrides


def test_checkpoint_feasibility_totals_actions_and_success(tmp_path: Path) -> None:
    config = resolved_config(overrides=knapsack_overrides(tmp_path))
    reproduce_pilot(config)
    paths = PilotPaths.from_config(config)
    originals = {
        name: read_parquet(paths.raw / f"{name}.parquet", name)
        for name in ("instances", "trials", "checkpoints")
    }
    checkpoints = originals["checkpoints"].to_pylist()

    def issues_for(*, points=None) -> set[str]:
        tables = dict(originals)
        if points is not None:
            tables["checkpoints"] = pa.Table.from_pylist(points, schema=get_schema("checkpoints"))
        issues = []
        validate_knapsack_dataset(paths.raw, tables, issues)
        return {issue.code for issue in issues}

    assert issues_for() == set()
    first = checkpoints[0]
    state = json.loads(first["state_json"])
    changed = [dict(row) for row in checkpoints]
    state["total_weight"] += 1
    changed[0]["state_json"] = json.dumps(state)
    assert "checkpoint_total_weight_mismatch" in issues_for(points=changed)

    state = json.loads(first["state_json"])
    changed = [dict(row) for row in checkpoints]
    state["total_value"] += 1
    changed[0]["state_json"] = json.dumps(state)
    assert "checkpoint_total_value_mismatch" in issues_for(points=changed)

    state = json.loads(first["state_json"])
    changed = [dict(row) for row in checkpoints]
    state["selected"] = [1] * len(state["selected"])
    state["total_weight"] = sum(state["weights"])
    state["total_value"] = sum(state["values"])
    changed[0]["state_json"] = json.dumps(state)
    assert "checkpoint_infeasible" in issues_for(points=changed)

    changed = [dict(row) for row in checkpoints]
    changed[1]["state_json"] = first["state_json"]
    changed[1]["objective_value"] = first["objective_value"]
    changed[1]["optimality_gap"] = first["optimality_gap"]
    assert "checkpoint_action_mismatch" in issues_for(points=changed)

    successful = next(row for row in originals["trials"].to_pylist() if row["success"])
    same_trial = [row for row in checkpoints if row["trial_id"] == successful["trial_id"]]
    initial = min(same_trial, key=lambda row: row["checkpoint_index"])
    final = max(same_trial, key=lambda row: row["checkpoint_index"])
    changed = [dict(row) for row in checkpoints]
    replacement = next(row for row in changed if row["checkpoint_id"] == final["checkpoint_id"])
    replacement["state_json"] = initial["state_json"]
    replacement["objective_value"] = initial["objective_value"]
    replacement["optimality_gap"] = initial["optimality_gap"]
    assert "trial_success_reference_mismatch" in issues_for(points=changed)
