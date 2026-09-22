"""A small scientific golden case excluding timestamps, runtime and run ID."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from llm_search_dynamics.config import resolved_config
from llm_search_dynamics.data.splits import read_split
from llm_search_dynamics.pipeline.stages import PilotPaths, reproduce_pilot


def test_default_seed_state_transition_metric_and_report_shape(tmp_path: Path) -> None:
    config = resolved_config(
        overrides=[
            "experiment=pilot",
            "task=dummy_binary",
            "solver=none",
            "search_method=randomized_first_improvement",
            "search.budget_limit=120",
            "state_model=baseline",
            "storage=local",
            "tracking=local",
            f"storage.raw_data_dir={tmp_path / 'raw'}",
            f"storage.interim_data_dir={tmp_path / 'interim'}",
            f"storage.derived_data_dir={tmp_path / 'derived'}",
            f"storage.artifact_dir={tmp_path / 'artifacts'}",
            f"tracking.tracking_uri=file:{tmp_path / 'mlruns'}",
        ]
    )
    reproduce_pilot(config)
    paths = PilotPaths.from_config(config)
    instances = pq.read_table(paths.raw / "instances.parquet").to_pylist()
    assert instances[0]["instance_id"] == "ins_563df6cd932456a53401281a"
    assert read_split(paths.derived / "splits.json").split_hash == (
        "0a0f3ad758bdacfc0df3f2ed4d63aec38448cb471da07f0d8fb702757c0d0efa"
    )
    states = json.loads((paths.derived / "state_assignments.json").read_text(encoding="utf-8"))
    assert states[0]["discrete_state"] == 3
    matrix = json.loads((paths.artifacts / "dynamics_model.json").read_text(encoding="utf-8"))[
        "transition_matrix"
    ]
    assert matrix[0] == [1, 0, 0, 0, 0, 0, 0]
    assert matrix[1][0] == pytest.approx(0.7777777777777778)
    assert matrix[2][1] == pytest.approx(0.7777777777777778)
    metrics = {
        row["metric_name"]: row
        for row in pq.read_table(paths.derived / "metrics.parquet").to_pylist()
    }
    assert metrics["one_step_nll"]["metric_value"] == pytest.approx(0.53775348, abs=1e-7)
    assert metrics["one_step_accuracy"]["metric_value"] == pytest.approx(12 / 14)
    assert metrics["success_brier_score"]["metric_value"] == pytest.approx(0.0183217, abs=1e-7)
    assert metrics["one_step_nll"]["n_units"] == 14
    with (paths.artifacts / "summary.csv").open(encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    assert columns == ["metric", "value"]
