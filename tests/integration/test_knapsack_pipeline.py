"""Isolated Phase 3 CPU integration across storage, solver, models and tracking."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pyarrow.parquet as pq
from typer.testing import CliRunner

from combinatorial_search_dynamics.cli import app
from combinatorial_search_dynamics.config import resolved_config
from combinatorial_search_dynamics.data.duckdb import create_analysis_views
from combinatorial_search_dynamics.data.splits import read_split
from combinatorial_search_dynamics.data.validation import validate_dataset
from combinatorial_search_dynamics.pipeline.stages import PilotPaths


def knapsack_overrides(directory: Path) -> list[str]:
    return [
        "experiment=knapsack_pilot",
        "task=knapsack",
        "solver=ortools_cp_sat",
        "search_method=randomized_first_improvement",
        "search.budget_limit=120",
        "state_model=objective_gap",
        "storage=knapsack_local",
        "tracking=knapsack_local",
        "experiment.instance_count=6",
        "search.trials=2",
        f"storage.raw_data_dir={directory / 'raw'}",
        f"storage.interim_data_dir={directory / 'interim'}",
        f"storage.derived_data_dir={directory / 'derived'}",
        f"storage.artifact_dir={directory / 'artifacts'}",
        f"storage.reference_table_path={directory / 'raw' / 'reference_solutions.parquet'}",
        f"tracking.tracking_uri=file:{directory / 'mlruns'}",
    ]


def test_knapsack_cli_pipeline_and_phase_one_validation(tmp_path: Path) -> None:
    options = knapsack_overrides(tmp_path)
    config = resolved_config(overrides=options)
    paths = PilotPaths.from_config(config)
    invocation = CliRunner().invoke(app, ["generate-instances", *options])
    assert invocation.exit_code == 0, invocation.output
    solve = CliRunner().invoke(app, ["solve-references", *options])
    assert solve.exit_code == 0 and "OPTIMAL" in solve.output
    assert CliRunner().invoke(app, ["solve-references", *options]).exit_code == 0
    for stage in (
        "collect",
        "prepare",
        "extract",
        "split",
        "analyze-success",
        "fit",
        "evaluate",
        "report",
    ):
        result = CliRunner().invoke(app, [stage, *options])
        assert result.exit_code == 0, (stage, result.output)
    assert validate_dataset(paths.raw).is_valid
    assert CliRunner().invoke(app, ["validate-data", "--data-dir", str(paths.raw)]).exit_code == 0
    instances = pq.read_table(paths.raw / "instances.parquet").to_pylist()
    references = pq.read_table(paths.reference).to_pylist()
    trials = pq.read_table(paths.raw / "trials.parquet").to_pylist()
    checkpoints = pq.read_table(paths.raw / "checkpoints.parquet").to_pylist()
    assert len(instances) == len(references) == 6
    assert all(row["solver_status"] == "OPTIMAL" for row in references)
    assert all(row["optimality_gap"] is not None for row in checkpoints)
    assert all(
        row["objective_value"]
        <= next(
            ref["optimal_value"]
            for ref in references
            if ref["instance_id"]
            == next(
                trial["instance_id"] for trial in trials if trial["trial_id"] == row["trial_id"]
            )
        )
        for row in checkpoints
    )
    split = read_split(paths.derived / "splits.json")
    train = set(split.instance_ids("train"))
    test = set(split.instance_ids("test"))
    for artifact in ("state_model.json", "dynamics_model.json"):
        model = json.loads((paths.artifacts / artifact).read_text(encoding="utf-8"))
        assert set(model["fit_metadata"]["instance_ids"]) <= train
        assert test.isdisjoint(model["fit_metadata"]["instance_ids"])
    run_id = json.loads((paths.artifacts / "run.json").read_text(encoding="utf-8"))["run_id"]
    metrics = pq.read_table(paths.derived / "metrics.parquet").to_pylist()
    assert {row["run_id"] for row in metrics} == {run_id}
    assert "knapsack_final_value_mean" in {row["metric_name"] for row in metrics}
    client = mlflow.tracking.MlflowClient(tracking_uri=paths.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["solver_name"] == "ortools_cp_sat"
    assert run.data.tags["reference_schema_version"] == "2"
    assert {entry.path for entry in client.list_artifacts(run_id)} >= {
        "reference_solutions.parquet",
        "solver_config.json",
        "solver_status_summary.json",
        "feasible_check.json",
        "report.md",
        "provenance.json",
    }
    connection = create_analysis_views(paths.raw)
    assert connection.execute("select count(*) from analysis_checkpoints").fetchone()[0] == len(
        checkpoints
    )
    connection.close()
    assert "OR-Tools" in (paths.artifacts / "report.md").read_text(encoding="utf-8")
    assert run.data.tags["search_method_name"] == "randomized_first_improvement"
    selection = json.loads(
        (paths.artifacts / "success_feature_selection.json").read_text(encoding="utf-8")
    )
    assert set(selection["selection"]) == {"randomized_first_improvement"}
    collision = CliRunner().invoke(app, ["generate-instances", *options])
    assert collision.exit_code != 0 and "existing output" in collision.output


def test_knapsack_science_repeats_for_same_seed(tmp_path: Path) -> None:
    results = []
    for name in ("a", "b"):
        options = knapsack_overrides(tmp_path / name)
        assert CliRunner().invoke(app, ["reproduce-pilot", *options]).exit_code == 0
        paths = PilotPaths.from_config(resolved_config(overrides=options))
        results.append(
            (
                [
                    (row["instance_id"], row["instance_json"])
                    for row in pq.read_table(paths.raw / "instances.parquet").to_pylist()
                ],
                [
                    (row["instance_id"], row["optimal_value"])
                    for row in pq.read_table(paths.reference).to_pylist()
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
    assert results[0] == results[1]


def test_three_methods_are_paired_and_success_models_are_method_specific(tmp_path: Path) -> None:
    options = [
        item
        for item in knapsack_overrides(tmp_path)
        if not item.startswith(
            ("search_method=", "search.budget_limit=", "experiment.instance_count=")
        )
    ]
    options.extend(
        [
            "search_method=classical_comparison",
            "search.budget_limit=6",
            "experiment.instance_count=9",
            "features.instance_sampling.sample_count=8",
            "features.instance_sampling.random_walk_steps=8",
            "evaluation.success_prediction.bootstrap_samples=20",
        ]
    )
    result = CliRunner().invoke(app, ["reproduce-pilot", *options])
    assert result.exit_code == 0, result.output
    paths = PilotPaths.from_config(resolved_config(overrides=options))
    trials = pq.read_table(paths.raw / "trials.parquet").to_pylist()
    paired: dict[tuple[str, int], list[dict[str, object]]] = {}
    for trial in trials:
        paired.setdefault((trial["instance_id"], trial["search_seed"]), []).append(trial)
    expected_methods = {
        "randomized_first_improvement",
        "simulated_annealing",
        "short_term_tabu",
    }
    assert all(
        {row["search_method_name"] for row in group} == expected_methods
        for group in paired.values()
    )
    assert all(len({row["initial_state_json"] for row in group}) == 1 for group in paired.values())
    comparison = json.loads(
        (paths.artifacts / "success_feature_comparison.json").read_text(encoding="utf-8")
    )
    assert set(comparison["comparison"]) == expected_methods
    assert pq.read_table(paths.derived / "instance_features.parquet").num_rows == 9
