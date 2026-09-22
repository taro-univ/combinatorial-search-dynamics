"""CPU pilot stages with explicit inputs, outputs, and collision checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from combinatorial_search_dynamics.config import repository_root
from combinatorial_search_dynamics.data.parquet import read_parquet, write_parquet
from combinatorial_search_dynamics.data.reference_validation import validate_references
from combinatorial_search_dynamics.data.references import reference_id, reference_row
from combinatorial_search_dynamics.data.schemas import SCHEMA_VERSION, empty_table, get_schema
from combinatorial_search_dynamics.data.splits import make_instance_split, read_split, write_split
from combinatorial_search_dynamics.data.validation import validate_dataset
from combinatorial_search_dynamics.data.zarr import ObservationArray, write_observation_store
from combinatorial_search_dynamics.dynamics.transition import MarkovTransitionModel
from combinatorial_search_dynamics.evaluation.knapsack import evaluate_knapsack
from combinatorial_search_dynamics.evaluation.metrics import evaluate_trajectories
from combinatorial_search_dynamics.evaluation.success_features import run_feature_selection
from combinatorial_search_dynamics.features.external import extract_external_features
from combinatorial_search_dynamics.features.success import (
    build_instance_feature_rows,
    build_success_feature_rows,
)
from combinatorial_search_dynamics.identifiers import (
    canonical_json,
    checkpoint_id,
    experiment_id,
    instance_id,
    trial_id,
)
from combinatorial_search_dynamics.provenance import collect_provenance, config_hash
from combinatorial_search_dynamics.reporting.tables import build_pilot_report
from combinatorial_search_dynamics.search.base import BudgetedEvaluator
from combinatorial_search_dynamics.search.initialization import make_initial_state
from combinatorial_search_dynamics.search.registry import get_search_algorithms
from combinatorial_search_dynamics.solvers.base import SolverParameters, SolverResult, SolverStatus
from combinatorial_search_dynamics.solvers.registry import get_solver
from combinatorial_search_dynamics.state_models.registry import get_state_model, load_state_model
from combinatorial_search_dynamics.tasks.dummy_binary import BinaryInstance
from combinatorial_search_dynamics.tasks.knapsack import KnapsackInstance
from combinatorial_search_dynamics.tasks.registry import get_task
from combinatorial_search_dynamics.tracking.mlflow import (
    local_tracking_uri,
    log_artifacts,
    record_run,
)


@dataclass(frozen=True)
class PilotPaths:
    root: Path
    raw: Path
    interim: Path
    derived: Path
    artifacts: Path
    reference: Path | None
    tracking_uri: str

    @classmethod
    def from_config(cls, config: dict[str, Any], *, root: Path | None = None) -> PilotPaths:
        source = (root or repository_root()).resolve()

        def resolve(value: str) -> Path:
            path = Path(value)
            return (path if path.is_absolute() else source / path).resolve()

        storage = config["storage"]
        return cls(
            root=source,
            raw=resolve(storage["raw_data_dir"]),
            interim=resolve(storage["interim_data_dir"]),
            derived=resolve(storage["derived_data_dir"]),
            artifacts=resolve(storage["artifact_dir"]),
            reference=resolve(storage["reference_table_path"])
            if storage.get("reference_table_path")
            else None,
            tracking_uri=local_tracking_uri(config["tracking"]["tracking_uri"], root=source),
        )


def _json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required preceding stage artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _outputs(paths: PilotPaths, stage: str, config: dict[str, Any]) -> list[Path]:
    mapping = {
        "generate-instances": [paths.raw / "instances.parquet"],
        "solve-references": [paths.reference] if paths.reference is not None else [],
        "collect": [
            paths.raw / "trials.parquet",
            paths.raw / "checkpoints.parquet",
            paths.raw / "observations.zarr",
            paths.raw / "metrics.parquet",
        ],
        "prepare": [paths.interim / "trajectories.json"],
        "extract": [paths.derived / "features.json"],
        "split": [paths.derived / "splits.json"],
        "analyze-success": [
            paths.derived / "success_features.parquet",
            paths.derived / "instance_features.parquet",
            paths.derived / "success_predictions.parquet",
            paths.artifacts / "success_feature_models.json",
            paths.artifacts / "success_feature_selection.json",
            paths.artifacts / "success_feature_comparison.json",
            paths.artifacts / "success_feature_comparison.md",
            paths.artifacts / "success_analysis_config.json",
        ],
        "fit": [
            paths.artifacts / "state_model.json",
            paths.artifacts / "dynamics_model.json",
            paths.derived / "state_assignments.json",
        ],
        "evaluate": [
            paths.derived / "metrics.parquet",
            paths.derived / "predictions.json",
            paths.derived / "evaluation_summary.json",
            paths.artifacts / "provenance.json",
            paths.artifacts / "resolved_config.json",
            paths.artifacts / "validation.json",
            paths.artifacts / "run.json",
        ],
        "report": [paths.artifacts / "report.md", paths.artifacts / "summary.csv"],
    }
    outputs = mapping[stage]
    if stage == "evaluate" and config["task"]["name"] == "knapsack":
        outputs = [
            *outputs,
            paths.artifacts / "solver_config.json",
            paths.artifacts / "solver_status_summary.json",
            paths.artifacts / "feasible_check.json",
        ]
    return outputs


def _guard_outputs(
    paths: PilotPaths, stage: str, config: dict[str, Any], *, force: bool
) -> list[Path]:
    outputs = _outputs(paths, stage, config)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Stage {stage} has existing output(s): {', '.join(str(path) for path in existing)}"
        )
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return outputs


def _table(name: str, rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=get_schema(name)) if rows else empty_table(name)


def _task(config: dict[str, Any]) -> Any:
    params = {
        key: value
        for key, value in config["task"].items()
        if key not in {"name", "version", "implemented"}
    }
    return get_task(config["task"]["name"], max_steps=_configured_budget_limit(config), **params)


def _configured_budget_limit(config: dict[str, Any]) -> int:
    configured = config["search"].get("budget_limit")
    if configured is not None:
        limit = int(configured)
    else:
        size = int(config["task"].get("item_count", config["task"].get("bit_count", 1)))
        limit = int(config["search"]["budget_multiplier"]) * size
    if limit <= 0:
        raise ValueError("search budget_limit must be positive")
    return limit


def _instance_from_row(task: Any, row: dict[str, Any]) -> BinaryInstance | KnapsackInstance:
    payload = json.loads(row["instance_json"])
    if task.name == "knapsack":
        instance = KnapsackInstance.from_dict(payload)
    else:
        instance = BinaryInstance.from_dict(payload)
    task.initial_state(instance)
    return instance


def _solver_parameters(config: dict[str, Any]) -> SolverParameters:
    selected = config["solver"]
    return SolverParameters(
        max_time_seconds=float(selected["max_time_seconds"]),
        num_search_workers=selected["num_search_workers"],
        random_seed=selected["random_seed"],
        relative_gap_epsilon=float(selected["relative_gap_epsilon"]),
        log_search_progress=selected["log_search_progress"],
    )


def _conditions(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": config["task"],
        "search": config["search"],
        "search_method": config["search_method"],
        "experiment_name": config["experiment"]["name"],
    }


def _generate_instances(config: dict[str, Any], paths: PilotPaths) -> None:
    task = _task(config)
    count = int(config["experiment"]["instance_count"])
    if count < 3:
        raise ValueError("At least three instances are required for train/validation/test")
    base_seed = int(config["experiment"]["seed"]["instance_generation"])
    rows: list[dict[str, Any]] = []
    for index in range(count):
        seed = base_seed + index
        instance = task.generate_instance(seed)
        identifier = instance_id(
            {
                "task_name": task.name,
                "task_version": task.version,
                "generation_seed": seed,
                "instance": instance.to_dict(),
            }
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "instance_id": identifier,
                "task_name": task.name,
                "task_version": task.version,
                "problem_size": instance.item_count
                if task.name == "knapsack"
                else instance.bit_count,
                "difficulty_value": None
                if task.name == "knapsack"
                else float(
                    sum(a != b for a, b in zip(instance.initial_bits, instance.target_bits))
                ),
                "generation_seed": seed,
                "instance_json": canonical_json(instance.to_dict()),
                # Stable fixture timestamp: wall-clock time cannot influence raw science.
                "created_at": datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=seed),
            }
        )
    write_parquet(_table("instances", rows), paths.raw / "instances.parquet", "instances")


def _solve_references(config: dict[str, Any], paths: PilotPaths) -> None:
    if config["task"]["name"] != "knapsack" or paths.reference is None:
        raise ValueError("solve-references requires the knapsack task and reference path")
    task = _task(config)
    parameters = _solver_parameters(config)
    solver = get_solver(config["solver"]["name"], task=task, parameters=parameters)
    instances = read_parquet(paths.raw / "instances.parquet", "instances")
    rows: list[dict[str, Any]] = []
    for source in instances.to_pylist():
        try:
            instance = _instance_from_row(task, source)
            result = solver.solve(instance, instance_id=source["instance_id"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            result = SolverResult(
                instance_id=source["instance_id"],
                solver_name=solver.name,
                solver_version=solver.version,
                status=SolverStatus.ERROR,
                best_feasible_value=None,
                best_bound=None,
                optimal_value=None,
                optimality_gap=None,
                optimality_proven=False,
                timed_out=False,
                runtime_seconds=0.0,
                solution=None,
                parameters=parameters,
                error_type=type(exc).__name__,
            )
        rows.append(reference_row(result, task_name=task.name, task_version=task.version))
    table = _table("reference_solutions", rows)
    problems: list[Any] = []
    validate_references(table, instances, problems)
    if problems:
        raise ValueError(f"Reference table has {len(problems)} semantic validation errors")
    write_parquet(table, paths.reference, "reference_solutions")


def _reference_rows(paths: PilotPaths) -> dict[str, dict[str, Any]]:
    if paths.reference is None:
        raise ValueError("Reference path is not configured")
    table = read_parquet(paths.reference, "reference_solutions")
    return {row["instance_id"]: row for row in table.to_pylist()}


def _collect(config: dict[str, Any], paths: PilotPaths) -> None:
    task = _task(config)
    instances = read_parquet(paths.raw / "instances.parquet", "instances").to_pylist()
    algorithms = get_search_algorithms(config["search_method"])
    budget_limit = _configured_budget_limit(config)
    conditions = _conditions(config)
    exp_id = experiment_id(config["experiment"]["name"], conditions)
    trials: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    bits: list[list[int]] = []
    references = _reference_rows(paths) if task.name == "knapsack" else {}
    search_seed_base = int(config["experiment"]["seed"]["search"])
    initial_seed_base = int(config["experiment"]["seed"]["initial_state"])
    trial_count = int(config["search"]["trials"])
    if trial_count < 1:
        raise ValueError("search.trials must be positive")
    for instance_index, row in enumerate(instances):
        instance = _instance_from_row(task, row)
        reference = references.get(row["instance_id"])
        if task.name == "knapsack" and reference is None:
            raise ValueError(f"Reference is missing for instance {row['instance_id']}")
        optimal = (
            reference["optimal_value"] if reference and reference["optimality_proven"] else None
        )
        for trial_index in range(trial_count):
            offset = instance_index * trial_count + trial_index
            search_seed = search_seed_base + offset
            initial_seed = initial_seed_base + offset
            initial_state = make_initial_state(
                task=task,
                instance=instance,
                seed=initial_seed,
                strategy=str(config["search"]["initial_state"]["strategy"]),
                include_probability=float(config["search"]["initial_state"]["include_probability"]),
            )
            for algorithm in algorithms:
                evaluator = BudgetedEvaluator(task=task, budget_limit=budget_limit)
                result = algorithm.run(
                    task=task,
                    initial_state=initial_state,
                    evaluator=evaluator,
                    search_seed=search_seed,
                    reference_value=optimal,
                )
                method_conditions = {
                    **conditions,
                    "active_search_method": {
                        "name": algorithm.name,
                        "revision": algorithm.revision,
                        "parameters": algorithm.parameters(),
                    },
                }
                identifier = trial_id(row["instance_id"], method_conditions, search_seed)
                trials.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "trial_id": identifier,
                        "instance_id": row["instance_id"],
                        "experiment_id": exp_id,
                        "search_method_name": algorithm.name,
                        "search_method_revision": algorithm.revision,
                        "search_seed": search_seed,
                        "budget_type": str(config["search"]["budget_type"]),
                        "budget_limit": budget_limit,
                        "search_parameters_json": canonical_json(algorithm.parameters()),
                        "initial_state_json": canonical_json(task.serialize_state(initial_state)),
                        "terminal_class": result.terminal_class,
                        "success": result.success,
                        "runtime_seconds": result.runtime_seconds,
                        "status": result.status,
                        "error_type": result.error_type,
                    }
                )
                for point in result.checkpoints:
                    if not task.is_feasible(point.state):
                        raise ValueError(f"Search checkpoint is infeasible in trial {identifier}")
                    point_id = checkpoint_id(identifier, point.checkpoint_index)
                    checkpoints.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "checkpoint_id": point_id,
                            "trial_id": identifier,
                            "budget_used": point.budget_used,
                            "checkpoint_index": point.checkpoint_index,
                            "decision_step": point.decision_step,
                            "accepted_moves": point.accepted_moves,
                            "rejected_moves": point.rejected_moves,
                            "action_json": canonical_json(
                                {"bit_index": point.action, "accepted": point.action_accepted}
                            )
                            if point.action is not None
                            else None,
                            "state_json": canonical_json(task.serialize_state(point.state)),
                            "objective_value": point.objective,
                            "optimality_gap": task.objective_gap(
                                point.objective,
                                optimal,
                                epsilon=float(config["solver"]["relative_gap_epsilon"]),
                            )
                            if optimal is not None
                            else None,
                            "remaining_budget": point.remaining_budget,
                            "is_terminal": point.is_terminal,
                            "tensor_ref": None,
                        }
                    )
                    bits.append(
                        list(point.state.selected)
                        if task.name == "knapsack"
                        else list(point.state.bits)
                    )

    write_parquet(_table("trials", trials), paths.raw / "trials.parquet", "trials")
    write_parquet(
        _table("checkpoints", checkpoints), paths.raw / "checkpoints.parquet", "checkpoints"
    )
    values = np.asarray(bits, dtype=np.int8)
    write_observation_store(
        paths.raw / "observations.zarr",
        {
            "external_state": ObservationArray(
                values=values,
                valid_mask=np.ones_like(values, dtype=np.bool_),
                axis_names=("checkpoint", "item" if task.name == "knapsack" else "bit"),
            )
        },
        trial_ids=[row["trial_id"] for row in checkpoints],
        checkpoint_ids=[row["checkpoint_id"] for row in checkpoints],
        budget_used=[row["budget_used"] for row in checkpoints],
        search_method_revision=str(config["search_method"]["revision"]),
        observation_code_version=str(config["observation"]["code_version"]),
        observation_metadata={
            "source": "classical_search",
            "observation_kind": "external-only",
            "task_name": task.name,
        }
        if task.name == "knapsack"
        else {"source": "classical_search", "observation_kind": "external-only"},
    )
    # Last output completes the four-table Phase 1 dataset snapshot.
    write_parquet(empty_table("metrics"), paths.raw / "metrics.parquet", "metrics")
    result = validate_dataset(paths.raw)
    if not result.is_valid:
        raise ValueError(f"Collected dataset has {result.error_count} validation errors")


def _prepare(config: dict[str, Any], paths: PilotPaths) -> None:
    result = validate_dataset(paths.raw)
    if not result.is_valid:
        raise ValueError(f"Raw Phase 1 dataset has {result.error_count} validation errors")
    task = _task(config)
    trial_rows = read_parquet(paths.raw / "trials.parquet", "trials").to_pylist()
    checkpoint_rows = read_parquet(paths.raw / "checkpoints.parquet", "checkpoints").to_pylist()
    by_trial: dict[str, list[dict[str, Any]]] = {row["trial_id"]: [] for row in trial_rows}
    trial_instance = {row["trial_id"]: row["instance_id"] for row in trial_rows}
    for checkpoint in checkpoint_rows:
        try:
            parsed = task.deserialize_state(json.loads(checkpoint["state_json"]))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse checkpoint {checkpoint['checkpoint_id']}") from exc
        by_trial[checkpoint["trial_id"]].append(
            {
                "instance_id": trial_instance[checkpoint["trial_id"]],
                "trial_id": checkpoint["trial_id"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_index": checkpoint["checkpoint_index"],
                "state": task.serialize_state(parsed),
                "objective_value": checkpoint["objective_value"],
                "optimality_gap": checkpoint["optimality_gap"],
            }
        )
    trajectories = [
        {
            "instance_id": row["instance_id"],
            "trial_id": row["trial_id"],
            "checkpoints": sorted(by_trial[row["trial_id"]], key=lambda p: p["checkpoint_index"]),
        }
        for row in trial_rows
    ]
    _write_json(paths.interim / "trajectories.json", trajectories)


def _extract(paths: PilotPaths) -> None:
    trajectories = _json(paths.interim / "trajectories.json")
    features = extract_external_features(
        [checkpoint for trajectory in trajectories for checkpoint in trajectory["checkpoints"]]
    )
    _write_json(paths.derived / "features.json", features)


def _split(config: dict[str, Any], paths: PilotPaths) -> None:
    _json(paths.derived / "features.json")
    instances = read_parquet(paths.raw / "instances.parquet", "instances").to_pylist()
    split = make_instance_split(
        [row["instance_id"] for row in instances],
        seed=int(config["experiment"]["seed"]["data_split"]),
        ratios=config["evaluation"]["split_ratios"],
    )
    write_split(paths.derived / "splits.json", split)


def _write_analysis_parquet(path: Path, rows: list[dict[str, Any]], analysis_version: str) -> None:
    if not rows:
        raise ValueError(f"Analysis table cannot be empty: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"analysis_version": analysis_version.encode("ascii")}
    )
    pq.write_table(table, path, compression="zstd")


def _analyze_success(config: dict[str, Any], paths: PilotPaths) -> None:
    if config["task"]["name"] != "knapsack":
        raise ValueError("B2/M2/M3 success analysis is implemented only for knapsack")
    split = read_split(paths.derived / "splits.json")
    instances = read_parquet(paths.raw / "instances.parquet", "instances").to_pylist()
    trials = read_parquet(paths.raw / "trials.parquet", "trials").to_pylist()
    checkpoints = read_parquet(paths.raw / "checkpoints.parquet", "checkpoints").to_pylist()
    references = list(_reference_rows(paths).values())
    feature_config = config["features"]
    analysis_version = str(feature_config["analysis_version"])
    epsilon = float(feature_config["normalization_epsilon"])
    sample_config = feature_config["instance_sampling"]
    feature_rows = build_success_feature_rows(
        instances=instances,
        trials=trials,
        checkpoints=checkpoints,
        references=references,
        assignments=split.assignments,
        history_window_budget=int(feature_config["history_window_budget"]),
        epsilon=epsilon,
        distance_seed=int(sample_config["seed"]),
    )
    instance_rows = build_instance_feature_rows(
        instances=instances,
        references=references,
        sample_seed=int(sample_config["seed"]),
        sample_count=int(sample_config["sample_count"]),
        random_walk_steps=int(sample_config["random_walk_steps"]),
        epsilon=epsilon,
    )
    for row in (*feature_rows, *instance_rows):
        row["analysis_version"] = analysis_version
    for row in instance_rows:
        row["calculation_version"] = analysis_version
    settings = {
        **config["evaluation"]["success_prediction"],
        "probability_floor": config["evaluation"]["probability_floor"],
    }
    models, selection, comparison, predictions = run_feature_selection(
        feature_rows,
        settings=settings,
        bootstrap_seed=int(config["experiment"]["seed"]["bootstrap"]),
    )
    for row in predictions:
        row["analysis_version"] = analysis_version
    _write_analysis_parquet(
        paths.derived / "success_features.parquet", feature_rows, analysis_version
    )
    _write_analysis_parquet(
        paths.derived / "instance_features.parquet", instance_rows, analysis_version
    )
    _write_analysis_parquet(
        paths.derived / "success_predictions.parquet", predictions, analysis_version
    )
    metadata = {
        "analysis_version": analysis_version,
        "split_hash": split.split_hash,
        "fit_split": "train",
        "selection_split": "validation",
        "test_final_evaluation_only": True,
    }
    _write_json(paths.artifacts / "success_feature_models.json", {**metadata, "models": models})
    _write_json(
        paths.artifacts / "success_feature_selection.json", {**metadata, "selection": selection}
    )
    _write_json(
        paths.artifacts / "success_feature_comparison.json",
        {**metadata, "comparison": comparison},
    )
    _write_json(
        paths.artifacts / "success_analysis_config.json",
        {
            "analysis_version": analysis_version,
            "features": feature_config,
            "evaluation": settings,
            "experiment_seeds": config["experiment"]["seed"],
        },
    )
    lines = [
        "# B2・M2・M3 success prediction",
        "",
        "Feature selection used train/validation only. Test was evaluated once after freezing the configuration.",
        "",
        "| Search method | Selected features | B2 Brier | Final Brier | Improvement | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, result in comparison.items():
        baseline = result["models"]["B2"]["test_metrics"]["brier"]
        final = result["models"]["selected_final"]["test_metrics"]["brier"]
        paired = result["paired_vs_B2"]
        chosen = selection[method]["selected_final_features"]
        lines.append(
            f"| {method} | {', '.join(chosen)} | {baseline:.6f} | {final:.6f} | "
            f"{paired['brier_improvement']:.6f} | "
            f"[{paired['ci_lower']:.6f}, {paired['ci_upper']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Positive improvement means a lower instance-macro Brier score than B2.",
            "Predictive improvement is not interpreted as a causal effect.",
            "",
        ]
    )
    report = paths.artifacts / "success_feature_comparison.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def _trajectories_from_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trial_id"], []).append(row)
    return [
        {
            "instance_id": ordered[0]["instance_id"],
            "trial_id": trial_id,
            "states": [row["discrete_state"] for row in ordered],
            "checkpoint_ids": [row["checkpoint_id"] for row in ordered],
        }
        for trial_id, points in sorted(grouped.items())
        if (ordered := sorted(points, key=lambda row: row["checkpoint_index"]))
    ]


def _fit(config: dict[str, Any], paths: PilotPaths) -> None:
    features = _json(paths.derived / "features.json")
    split = read_split(paths.derived / "splits.json")
    train_ids = split.fit_instance_ids("train")
    model = get_state_model(config["state_model"], config["task"])
    eligible = (
        [row for row in features if row["optimality_gap"] is not None]
        if config["task"]["name"] == "knapsack"
        else features
    )
    train_features = [row for row in eligible if row["instance_id"] in train_ids]
    model.fit(
        train_features,
        fit_split="train",
        split_hash=split.split_hash,
        instance_ids=train_ids,
    )
    transformed = model.transform(eligible)
    _write_json(paths.derived / "state_assignments.json", transformed)
    model.save(paths.artifacts / "state_model.json")
    train_trajectories = _trajectories_from_states(
        [row for row in transformed if row["instance_id"] in train_ids]
    )
    dynamics = MarkovTransitionModel(
        state_count=model.state_count if hasattr(model, "state_count") else model.bit_count + 1,
        smoothing=float(config["dynamics"]["smoothing"]),
    )
    dynamics.fit(
        [row["states"] for row in train_trajectories],
        fit_split="train",
        split_hash=split.split_hash,
        instance_ids=[row["instance_id"] for row in train_trajectories],
    )
    dynamics.save(paths.artifacts / "dynamics_model.json")


def _evaluate(config: dict[str, Any], paths: PilotPaths) -> None:
    started_at = datetime.now(UTC)
    if int(config["evaluation"]["horizon"]) != 1:
        raise ValueError("The pilot evaluation implements only the one-step horizon")
    split = read_split(paths.derived / "splits.json")
    state_model = load_state_model(paths.artifacts / "state_model.json")
    dynamics = MarkovTransitionModel.load(paths.artifacts / "dynamics_model.json")
    if state_model.fit_metadata["split_hash"] != split.split_hash:
        raise ValueError("State model split hash differs from the current split")
    if dynamics.fit_metadata["split_hash"] != split.split_hash:
        raise ValueError("Dynamics model split hash differs from the current split")
    test_ids = set(split.instance_ids("test"))
    for model in (state_model, dynamics):
        if model.fit_metadata["fit_split"] != "train":
            raise ValueError("Model was not fit using only train")
        if test_ids & set(model.fit_metadata["instance_ids"]):
            raise ValueError("Test instance leakage into model fit metadata")
    states = _json(paths.derived / "state_assignments.json")
    trajectories = _trajectories_from_states(
        [row for row in states if row["instance_id"] in test_ids]
    )
    result = evaluate_trajectories(
        dynamics,
        trajectories,
        probability_floor=float(config["evaluation"]["probability_floor"]),
    )
    if not result.n_transitions:
        raise ValueError("No held-out transitions with proven references are available")
    validation = validate_dataset(paths.raw)
    if not validation.is_valid:
        raise ValueError(f"Dataset validation failed with {validation.error_count} errors")
    problem = None
    problem_metric_values: dict[str, float] = {}
    extra_artifacts: list[Path] = []
    if config["task"]["name"] == "knapsack":
        task = _task(config)
        references = list(_reference_rows(paths).values())
        problem = evaluate_knapsack(
            task=task,
            instances=read_parquet(paths.raw / "instances.parquet", "instances").to_pylist(),
            trials=read_parquet(paths.raw / "trials.parquet", "trials").to_pylist(),
            checkpoints=read_parquet(paths.raw / "checkpoints.parquet", "checkpoints").to_pylist(),
            references=references,
            test_instance_ids=test_ids,
        )
        problem_metric_values = {name: metric.value for name, metric in problem.metrics.items()}
        solver_config_path = paths.artifacts / "solver_config.json"
        status_path = paths.artifacts / "solver_status_summary.json"
        feasible_path = paths.artifacts / "feasible_check.json"
        _write_json(solver_config_path, config["solver"])
        _write_json(status_path, problem.report)
        _write_json(
            feasible_path,
            {
                "valid": validation.is_valid,
                "checkpoint_issue_count": sum(
                    issue.table_or_artifact == "checkpoints" for issue in validation.issues
                ),
            },
        )
        extra_artifacts = [paths.reference, solver_config_path, status_path, feasible_path]
    _write_json(
        paths.derived / "predictions.json",
        {
            "predictions": result.predictions,
            "per_instance": result.per_instance,
        },
    )
    _write_json(paths.artifacts / "resolved_config.json", config)
    _write_json(
        paths.artifacts / "validation.json",
        {
            "valid": validation.is_valid,
            "error_count": validation.error_count,
            "issues": [issue.__dict__ for issue in validation.issues],
        },
    )
    provenance = collect_provenance(
        root=paths.root, config=config, started_at=started_at, ended_at=datetime.now(UTC)
    )
    if problem is not None:
        provenance["solver"] = {
            "ortools_version": provenance["dependency_versions"]["ortools"],
            "solver_class": "OrtoolsKnapsackSolver",
            "parameters": config["solver"],
            "seed": config["solver"]["random_seed"],
            "reference_table_sha256": hashlib.sha256(paths.reference.read_bytes()).hexdigest(),
            "reference_schema_version": SCHEMA_VERSION,
        }
    _write_json(paths.artifacts / "provenance.json", provenance)
    config_digest = config_hash(config)
    tags = {
        "git_commit": provenance["git_commit"],
        "code_status": provenance["code_status"],
        "config_hash": config_digest,
        "schema_version": SCHEMA_VERSION,
        "task_name": config["task"]["name"],
        "task_version": str(config["task"]["version"]),
        "split_hash": split.split_hash,
        "search_method_name": str(config["search_method"]["name"]),
        "search_method_revision": str(config["search_method"]["revision"]),
        "dvc_revision": provenance["dvc_revision"],
    }
    if problem is not None:
        tags.update(
            {
                "solver_name": config["solver"]["name"],
                "solver_version": provenance["dependency_versions"]["ortools"],
                "solver_status_summary": canonical_json(problem.report["solver_status_counts"]),
                "reference_schema_version": SCHEMA_VERSION,
                "reference_configuration_hash": config_hash(
                    {"task": config["task"], "solver": config["solver"]}
                ),
            }
        )

    def write_run_outputs(run_id: str) -> list[Path]:
        rows = [
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "split": "test",
                "fold": None,
                "horizon": int(config["evaluation"]["horizon"]),
                "metric_name": name,
                "metric_value": value,
                "n_units": result.n_transitions,
            }
            for name, value in result.metrics.items()
        ]
        rows.extend(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "split": "test",
                "fold": None,
                "horizon": None,
                "metric_name": name,
                "metric_value": float(value),
                "n_units": 1,
            }
            for name, value in (
                ("evaluated_transitions", result.n_transitions),
                ("evaluated_instances", result.n_instances),
                ("evaluated_trials", result.n_trials),
            )
        )
        if problem is not None:
            rows.extend(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "split": "test",
                    "fold": None,
                    "horizon": None,
                    "metric_name": name,
                    "metric_value": metric.value,
                    "n_units": metric.n_units,
                }
                for name, metric in problem.metrics.items()
            )
        metric_path = paths.derived / "metrics.parquet"
        write_parquet(_table("metrics", rows), metric_path, "metrics")
        run_path = paths.artifacts / "run.json"
        _write_json(run_path, {"run_id": run_id, "config_hash": config_digest})
        summary_path = paths.derived / "evaluation_summary.json"
        _write_json(
            summary_path,
            {
                "run_id": run_id,
                "config_hash": config_digest,
                **result.metrics,
                "n_transitions": result.n_transitions,
                "n_instances": result.n_instances,
                "n_trials": result.n_trials,
                "problem_metrics": problem_metric_values,
                "problem_report": problem.report if problem is not None else None,
            },
        )
        return [metric_path, run_path, summary_path]

    record_run(
        tracking_uri=paths.tracking_uri,
        experiment_name=config["tracking"]["experiment_name"],
        parameters=config,
        metrics={
            **result.metrics,
            **problem_metric_values,
            "evaluated_transitions": float(result.n_transitions),
        },
        tags=tags,
        artifacts=[
            paths.artifacts / "resolved_config.json",
            paths.derived / "splits.json",
            paths.artifacts / "state_model.json",
            paths.artifacts / "dynamics_model.json",
            paths.derived / "predictions.json",
            paths.artifacts / "provenance.json",
            paths.artifacts / "validation.json",
            *extra_artifacts,
        ],
        prepare_artifacts=write_run_outputs,
    )


def _report(config: dict[str, Any], paths: PilotPaths) -> None:
    summary = _json(paths.derived / "evaluation_summary.json")
    run = _json(paths.artifacts / "run.json")
    split = read_split(paths.derived / "splits.json")
    instances = read_parquet(paths.raw / "instances.parquet", "instances")
    trials = read_parquet(paths.raw / "trials.parquet", "trials").to_pylist()
    checkpoints = read_parquet(paths.raw / "checkpoints.parquet", "checkpoints").to_pylist()
    last = {
        row["trial_id"]: row for row in sorted(checkpoints, key=lambda row: row["checkpoint_index"])
    }
    distances = [float(last[row["trial_id"]]["objective_value"]) for row in trials]
    values: dict[str, Any] = {
        "problem_type": config["task"]["name"],
        "run_id": run["run_id"],
        "config_hash": summary["config_hash"],
        "task": f"{config['task']['name']}@{config['task']['version']}",
        "search_method": (
            f"{config['search_method']['name']}@{config['search_method']['revision']}"
        ),
        "instances": instances.num_rows,
        "trials": len(trials),
        "checkpoints": len(checkpoints),
        "train_instances": len(split.instance_ids("train")),
        "validation_instances": len(split.instance_ids("validation")),
        "test_instances": len(split.instance_ids("test")),
        "success_rate": sum(row["success"] for row in trials) / len(trials),
        "one_step_nll": summary["one_step_nll"],
        "one_step_accuracy": summary["one_step_accuracy"],
        "success_brier_score": summary["success_brier_score"],
        "evaluated_transitions": summary["n_transitions"],
        "evaluated_instances": summary["n_instances"],
        "evaluated_trials": summary["n_trials"],
    }
    if config["task"]["name"] == "knapsack":
        references = list(_reference_rows(paths).values())
        values.update(
            {
                "item_count": config["task"]["item_count"],
                "weight_range": f"{config['task']['min_weight']}..{config['task']['max_weight']}",
                "value_range": f"{config['task']['min_value']}..{config['task']['max_value']}",
                "capacity_ratio": config["task"]["capacity_ratio"],
                "solver": f"{config['solver']['name']}@{references[0]['solver_version']}",
                "solver_status_counts": canonical_json(
                    summary["problem_report"]["solver_status_counts"]
                ),
                "solver_timeout_count": summary["problem_report"]["solver_timeout_count"],
                "solver_optimality_proven_rate": summary["problem_report"][
                    "solver_optimality_proven_rate"
                ],
                "solver_runtime_mean_seconds": summary["problem_report"][
                    "solver_runtime_mean_seconds"
                ],
                "mean_final_total_value": sum(distances) / len(distances),
                **summary["problem_metrics"],
            }
        )
    else:
        values["mean_final_hamming_distance"] = sum(distances) / len(distances)
    artifact_paths = [
        Path(os.path.relpath(path, paths.artifacts))
        for path in (
            paths.derived / "metrics.parquet",
            paths.derived / "predictions.json",
            paths.derived / "splits.json",
            paths.artifacts / "state_model.json",
            paths.artifacts / "dynamics_model.json",
            paths.artifacts / "provenance.json",
            paths.artifacts / "resolved_config.json",
            paths.artifacts / "validation.json",
            paths.artifacts / "summary.csv",
        )
    ]
    if config["task"]["name"] == "knapsack":
        artifact_paths.extend(
            Path(os.path.relpath(path, paths.artifacts))
            for path in (
                paths.reference,
                paths.artifacts / "solver_config.json",
                paths.artifacts / "solver_status_summary.json",
                paths.artifacts / "feasible_check.json",
            )
        )
    report_path, summary_path = build_pilot_report(
        report_path=paths.artifacts / "report.md",
        summary_path=paths.artifacts / "summary.csv",
        summary=values,
        artifact_paths=artifact_paths,
    )
    log_artifacts(
        tracking_uri=paths.tracking_uri,
        run_id=run["run_id"],
        artifacts=[report_path, summary_path],
    )


STAGE_NAMES = (
    "generate-instances",
    "solve-references",
    "collect",
    "prepare",
    "extract",
    "split",
    "analyze-success",
    "fit",
    "evaluate",
    "report",
)


def _references_reusable(config: dict[str, Any], paths: PilotPaths) -> bool:
    """Reuse a completed reference table only for identical instance/solver inputs."""
    if paths.reference is None or not paths.reference.is_file():
        return False
    instances = read_parquet(paths.raw / "instances.parquet", "instances")
    references = read_parquet(paths.reference, "reference_solutions")
    problems: list[Any] = []
    validate_references(references, instances, problems)
    if problems:
        return False
    parameters = _solver_parameters(config)
    solver = get_solver(config["solver"]["name"], task=_task(config), parameters=parameters)
    rows = references.to_pylist()
    if len(rows) != instances.num_rows:
        return False
    return all(
        row["reference_id"]
        == reference_id(
            row["instance_id"],
            solver_name=solver.name,
            solver_version=solver.version,
            parameters=parameters.to_dict(),
            solve_seed=parameters.random_seed,
        )
        for row in rows
    )


def run_stage(
    stage: str, config: dict[str, Any], *, root: Path | None = None, force: bool = False
) -> list[Path]:
    """Run one independent stage after checking prerequisites and output collisions."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown pilot stage: {stage}")
    paths = PilotPaths.from_config(config, root=root)
    if stage == "solve-references" and not force and _references_reusable(config, paths):
        return [paths.reference]
    outputs = _guard_outputs(paths, stage, config, force=force)
    if stage == "generate-instances":
        _generate_instances(config, paths)
    elif stage == "solve-references":
        _solve_references(config, paths)
    elif stage == "collect":
        _collect(config, paths)
    elif stage == "prepare":
        _prepare(config, paths)
    elif stage == "extract":
        _extract(paths)
    elif stage == "split":
        _split(config, paths)
    elif stage == "analyze-success":
        _analyze_success(config, paths)
    elif stage == "fit":
        _fit(config, paths)
    elif stage == "evaluate":
        _evaluate(config, paths)
    elif stage == "report":
        _report(config, paths)
    return outputs


def reproduce_pilot(
    config: dict[str, Any], *, root: Path | None = None, force: bool = False
) -> dict[str, list[Path]]:
    """Run every stage once, without invoking a recursive CLI or DVC process."""
    stages = (
        STAGE_NAMES
        if config["task"]["name"] == "knapsack"
        else tuple(
            stage for stage in STAGE_NAMES if stage not in {"solve-references", "analyze-success"}
        )
    )
    return {stage: run_stage(stage, config, root=root, force=force) for stage in stages}
