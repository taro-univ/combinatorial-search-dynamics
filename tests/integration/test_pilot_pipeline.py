"""Isolated CPU pilot: CLI, Phase 1 storage, DuckDB, MLflow and leakage."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pyarrow.parquet as pq
from typer.testing import CliRunner

from llm_search_dynamics.cli import app
from llm_search_dynamics.config import resolved_config
from llm_search_dynamics.data.duckdb import create_analysis_views
from llm_search_dynamics.data.splits import read_split
from llm_search_dynamics.data.validation import validate_dataset
from llm_search_dynamics.data.zarr import read_observation_store
from llm_search_dynamics.pipeline.stages import PilotPaths, reproduce_pilot


def _overrides(directory: Path) -> list[str]:
    return [
        "experiment=pilot",
        "task=dummy_binary",
        "solver=none",
        "state_model=baseline",
        "storage=local",
        "tracking=local",
        "experiment.instance_count=9",
        "task.bit_count=6",
        "generation.trials=2",
        f"storage.raw_data_dir={directory / 'raw'}",
        f"storage.interim_data_dir={directory / 'interim'}",
        f"storage.derived_data_dir={directory / 'derived'}",
        f"storage.artifact_dir={directory / 'artifacts'}",
        f"tracking.tracking_uri=file:{directory / 'mlruns'}",
    ]


def test_cli_full_pilot_and_phase_one_contract(tmp_path: Path) -> None:
    options = _overrides(tmp_path)
    invocation = CliRunner().invoke(app, ["reproduce-pilot", *options])
    assert invocation.exit_code == 0, invocation.output
    paths = PilotPaths.from_config(resolved_config(overrides=options))
    assert validate_dataset(paths.raw).is_valid
    assert CliRunner().invoke(app, ["validate-data", "--data-dir", str(paths.raw)]).exit_code == 0
    assert read_observation_store(paths.raw / "observations.zarr").observation_metadata == {
        "source": "mock",
        "observation_kind": "external-only",
    }
    store = read_observation_store(paths.raw / "observations.zarr")
    assert set(store.observations) == {"external_state"}
    assert store.observations["external_state"].valid_mask.all()
    trials = pq.read_table(paths.raw / "trials.parquet").to_pylist()
    checkpoints = pq.read_table(paths.raw / "checkpoints.parquet").to_pylist()
    split = read_split(paths.derived / "splits.json")
    for trial in trials:
        assert trial["instance_id"] in split.assignments
    conn = create_analysis_views(paths.raw)
    count, distinct_trials = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT trial_id) FROM analysis_checkpoints"
    ).fetchone()
    assert count == len(checkpoints)
    assert distinct_trials == len(trials)
    conn.close()

    state = json.loads((paths.artifacts / "state_model.json").read_text(encoding="utf-8"))
    dynamics = json.loads((paths.artifacts / "dynamics_model.json").read_text(encoding="utf-8"))
    train = set(split.instance_ids("train"))
    test = set(split.instance_ids("test"))
    for model in (state, dynamics):
        assert set(model["fit_metadata"]["instance_ids"]) == train
        assert test.isdisjoint(model["fit_metadata"]["instance_ids"])
    np.testing.assert_allclose(np.asarray(dynamics["transition_matrix"]).sum(axis=1), 1)
    metric_rows = pq.read_table(paths.derived / "metrics.parquet").to_pylist()
    run_id = json.loads((paths.artifacts / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert {row["run_id"] for row in metric_rows} == {run_id}
    assert {row["split"] for row in metric_rows} == {"test"}
    client = mlflow.tracking.MlflowClient(tracking_uri=paths.tracking_uri)
    logged_run = client.get_run(run_id)
    assert logged_run.data.tags["split_hash"] == split.split_hash
    assert logged_run.data.tags["task_name"] == "dummy_binary"
    assert logged_run.data.tags["generator_name"] == "mock_binary_search"
    assert logged_run.data.metrics["one_step_accuracy"] >= 0
    names = {artifact.path for artifact in client.list_artifacts(run_id)}
    assert {
        "resolved_config.json",
        "splits.json",
        "state_model.json",
        "dynamics_model.json",
        "metrics.parquet",
        "summary.csv",
        "report.md",
        "provenance.json",
        "validation.json",
    } <= names
    assert "Test instances were excluded" in (paths.artifacts / "report.md").read_text(
        encoding="utf-8"
    )
    second = CliRunner().invoke(app, ["generate-instances", *options])
    assert second.exit_code != 0 and "existing output" in second.output


def test_reproduce_pilot_science_ignores_run_ids_and_runtime(tmp_path: Path) -> None:
    outcomes = []
    for label in ("one", "two"):
        config = resolved_config(overrides=_overrides(tmp_path / label))
        reproduce_pilot(config)
        paths = PilotPaths.from_config(config)
        outcomes.append(
            (
                [
                    row["instance_id"]
                    for row in pq.read_table(paths.raw / "instances.parquet").to_pylist()
                ],
                [
                    row["trial_id"]
                    for row in pq.read_table(paths.raw / "trials.parquet").to_pylist()
                ],
                read_split(paths.derived / "splits.json").split_hash,
                json.loads((paths.artifacts / "dynamics_model.json").read_text(encoding="utf-8"))[
                    "transition_matrix"
                ],
                [
                    (row["metric_name"], row["metric_value"])
                    for row in pq.read_table(paths.derived / "metrics.parquet").to_pylist()
                ],
            )
        )
    assert outcomes[0] == outcomes[1]
