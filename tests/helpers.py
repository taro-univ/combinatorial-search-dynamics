from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from combinatorial_search_dynamics.data.parquet import write_parquet
from combinatorial_search_dynamics.data.schemas import get_schema
from combinatorial_search_dynamics.data.zarr import ObservationArray, write_observation_store
from combinatorial_search_dynamics.identifiers import checkpoint_id, instance_id, trial_id


def make_tables() -> dict[str, pa.Table]:
    schema_version = "2"
    instance = instance_id({"weights": [1, 2], "name": "小規模"})
    trial_a = trial_id(instance, {"temperature": 0.0}, 10)
    trial_b = trial_id(instance, {"temperature": 0.0}, 11)
    checkpoint_a0 = checkpoint_id(trial_a, 0)
    checkpoint_a1 = checkpoint_id(trial_a, 8)
    checkpoint_b0 = checkpoint_id(trial_b, 0)
    created_at = datetime(2026, 9, 12, tzinfo=UTC)

    rows = {
        "instances": [
            {
                "schema_version": schema_version,
                "instance_id": instance,
                "task_name": "placeholder",
                "task_version": "1",
                "problem_size": 2,
                "difficulty_value": None,
                "generation_seed": 100,
                "instance_json": '{"name":"小規模","weights":[1,2]}',
                "created_at": created_at,
            }
        ],
        "trials": [
            {
                "schema_version": schema_version,
                "trial_id": trial,
                "instance_id": instance,
                "experiment_id": "exp_pilot_fixture",
                "search_method_name": "fixture_search",
                "search_method_revision": "1",
                "search_seed": seed,
                "budget_type": "candidate_evaluations",
                "budget_limit": 16,
                "search_parameters_json": "{}",
                "initial_state_json": "{}",
                "terminal_class": "success",
                "success": True,
                "runtime_seconds": 0.1,
                "status": "completed",
                "error_type": None,
            }
            for trial, seed in ((trial_a, 10), (trial_b, 11))
        ],
        "checkpoints": [
            {
                "schema_version": schema_version,
                "checkpoint_id": checkpoint,
                "trial_id": trial,
                "budget_used": token,
                "checkpoint_index": index,
                "decision_step": index,
                "accepted_moves": index,
                "rejected_moves": token - index,
                "action_json": None if index == 0 else '{"accepted":true,"bit_index":0}',
                "state_json": "{}",
                "objective_value": float(index),
                "optimality_gap": None,
                "remaining_budget": 16 - token,
                "is_terminal": terminal,
                "tensor_ref": f"observations.zarr#{checkpoint}",
            }
            for checkpoint, trial, token, index, terminal in (
                (checkpoint_a0, trial_a, 0, 0, False),
                (checkpoint_a1, trial_a, 8, 1, True),
                (checkpoint_b0, trial_b, 0, 0, True),
            )
        ],
        "metrics": [
            {
                "schema_version": schema_version,
                "run_id": "run_fixture",
                "split": split,
                "fold": None,
                "horizon": None,
                "metric_name": "count",
                "metric_value": value,
                "n_units": 1,
            }
            for split, value in (("validation", 1.0), ("test", 1.0))
        ],
    }
    return {
        name: pa.Table.from_pylist(table_rows, schema=get_schema(name, schema_version))
        for name, table_rows in rows.items()
    }


def make_observations(tables: dict[str, pa.Table]) -> dict[str, ObservationArray]:
    count = tables["checkpoints"].num_rows
    return {
        name: ObservationArray(
            values=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
            valid_mask=np.ones((count, 2), dtype=np.bool_),
            axis_names=("checkpoint", "feature"),
        )
        for name in ("hidden", "attention_summary", "mlp_update", "kv_summary")
    }


def write_dataset(path: Path, tables: dict[str, pa.Table] | None = None) -> dict[str, pa.Table]:
    dataset = tables or make_tables()
    for name, table in dataset.items():
        write_parquet(table, path / f"{name}.parquet", name)
    checkpoints = dataset["checkpoints"]
    write_observation_store(
        path / "observations.zarr",
        make_observations(dataset),
        trial_ids=checkpoints.column("trial_id").to_pylist(),
        checkpoint_ids=checkpoints.column("checkpoint_id").to_pylist(),
        budget_used=checkpoints.column("budget_used").to_pylist(),
        search_method_revision="fixture-search",
        observation_code_version="fixture-v1",
    )
    return dataset
