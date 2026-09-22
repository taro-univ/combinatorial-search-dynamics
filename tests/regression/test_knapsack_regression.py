"""Stable Phase 3 science outputs; runtime, timestamps and run IDs are excluded."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from llm_search_dynamics.config import resolved_config
from llm_search_dynamics.data.schemas import get_schema
from llm_search_dynamics.data.splits import read_split
from llm_search_dynamics.pipeline.stages import PilotPaths, reproduce_pilot
from tests.integration.test_knapsack_pipeline import knapsack_overrides


def test_knapsack_fixed_seed_objective_states_matrix_and_metrics(tmp_path: Path) -> None:
    config = resolved_config(overrides=knapsack_overrides(tmp_path))
    reproduce_pilot(config)
    paths = PilotPaths.from_config(config)
    instance = pq.read_table(paths.raw / "instances.parquet").to_pylist()[0]
    reference = pq.read_table(paths.reference)
    checkpoint = pq.read_table(paths.raw / "checkpoints.parquet").to_pylist()[0]
    states = json.loads((paths.derived / "state_assignments.json").read_text(encoding="utf-8"))
    matrix = json.loads((paths.artifacts / "dynamics_model.json").read_text(encoding="utf-8"))[
        "transition_matrix"
    ]
    metrics = {
        row["metric_name"]: row["metric_value"]
        for row in pq.read_table(paths.derived / "metrics.parquet").to_pylist()
    }
    assert instance["instance_id"] == "ins_e2a49a6df4531a68d241828b"
    assert reference.schema == get_schema("reference_solutions")
    assert reference.to_pylist()[0]["optimal_value"] == 26.0
    assert checkpoint["objective_value"] == 23.0
    assert checkpoint["optimality_gap"] == pytest.approx(3 / 26)
    assert states[0]["discrete_state"] == 2
    assert read_split(paths.derived / "splits.json").split_hash == (
        "53a0295823feaec35e703933bd836163e8569281107ae4bc93cb23ccf132c0ad"
    )
    assert matrix[0] == [1, 0, 0, 0, 0, 0]
    assert matrix[1] == pytest.approx([1 / 6] * 6)
    assert metrics["one_step_nll"] == pytest.approx(2.349751613751945)
    assert metrics["one_step_accuracy"] == 0.0
    assert metrics["success_brier_score"] == pytest.approx(0.3292824074074074)
    assert metrics["knapsack_final_value_mean"] == 24.0
    assert metrics["knapsack_success_rate_trial"] == 0.5
    with (paths.artifacts / "summary.csv").open(encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    assert columns == ["metric", "value"]
    report = (paths.artifacts / "report.md").read_text(encoding="utf-8")
    assert "Phase 3 CPU pilot report" in report
    assert "OR-Tools CP-SAT" in report
