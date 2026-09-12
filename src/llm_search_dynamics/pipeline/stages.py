"""Phase 2 pilot stages with explicit inputs, outputs, and collision checks."""

from __future__ import annotations

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

from llm_search_dynamics.config import repository_root
from llm_search_dynamics.data.parquet import read_parquet, write_parquet
from llm_search_dynamics.data.schemas import SCHEMA_VERSION, empty_table, get_schema
from llm_search_dynamics.data.splits import make_instance_split, read_split, write_split
from llm_search_dynamics.data.validation import validate_dataset
from llm_search_dynamics.data.zarr import ObservationArray, write_observation_store
from llm_search_dynamics.dynamics.transition import MarkovTransitionModel
from llm_search_dynamics.evaluation.metrics import evaluate_trajectories
from llm_search_dynamics.features.external import extract_external_features
from llm_search_dynamics.generation.mock import (
    GENERATOR_NAME,
    GENERATOR_REVISION,
    MockGenerator,
)
from llm_search_dynamics.identifiers import canonical_json, experiment_id, instance_id
from llm_search_dynamics.provenance import collect_provenance, config_hash
from llm_search_dynamics.reporting.tables import build_pilot_report
from llm_search_dynamics.state_models.baseline import HammingDistanceStateModel
from llm_search_dynamics.tasks.dummy_binary import BinaryInstance
from llm_search_dynamics.tasks.registry import get_task
from llm_search_dynamics.tracking.mlflow import local_tracking_uri, log_artifacts, record_run


@dataclass(frozen=True)
class PilotPaths:
    root: Path
    raw: Path
    interim: Path
    derived: Path
    artifacts: Path
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


def _outputs(paths: PilotPaths, stage: str) -> list[Path]:
    mapping = {
        "generate-instances": [paths.raw / "instances.parquet"],
        "collect": [
            paths.raw / "trials.parquet",
            paths.raw / "checkpoints.parquet",
            paths.raw / "observations.zarr",
            paths.raw / "metrics.parquet",
        ],
        "prepare": [paths.interim / "trajectories.json"],
        "extract": [paths.derived / "features.json"],
        "split": [paths.derived / "splits.json"],
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
    return mapping[stage]


def _guard_outputs(paths: PilotPaths, stage: str, *, force: bool) -> list[Path]:
    outputs = _outputs(paths, stage)
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
    return get_task(
        config["task"]["name"],
        bit_count=int(config["task"]["bit_count"]),
        max_steps=int(config["generation"]["max_steps"]),
    )


def _conditions(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": config["task"],
        "generator": config["generation"],
        "mock_identity": config["llm"],
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
                "problem_size": instance.bit_count,
                "difficulty_value": float(
                    sum(a != b for a, b in zip(instance.initial_bits, instance.target_bits))
                ),
                "generation_seed": seed,
                "instance_json": canonical_json(instance.to_dict()),
                # Stable fixture timestamp: wall-clock time cannot influence raw science.
                "created_at": datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=seed),
            }
        )
    write_parquet(_table("instances", rows), paths.raw / "instances.parquet", "instances")


def _collect(config: dict[str, Any], paths: PilotPaths) -> None:
    task = _task(config)
    instances = read_parquet(paths.raw / "instances.parquet", "instances").to_pylist()
    generator = MockGenerator(
        greedy_probability=float(config["generation"]["greedy_probability"]),
        max_steps=int(config["generation"]["max_steps"]),
    )
    conditions = _conditions(config)
    exp_id = experiment_id(config["experiment"]["name"], conditions)
    trials: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    bits: list[list[int]] = []
    seed_base = int(config["experiment"]["seed"]["llm_sampling"])
    trial_count = int(config["generation"]["trials"])
    if trial_count < 1:
        raise ValueError("generation.trials must be positive")
    for instance_index, row in enumerate(instances):
        instance = BinaryInstance.from_dict(json.loads(row["instance_json"]))
        task.initial_state(instance)
        for trial_index in range(trial_count):
            sampling_seed = seed_base + instance_index * trial_count + trial_index
            generated = generator.generate(
                task=task,
                instance=instance,
                source_instance_id=row["instance_id"],
                experiment_conditions=conditions,
                sampling_seed=sampling_seed,
            )
            trials.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "trial_id": generated.trial_id,
                    "instance_id": row["instance_id"],
                    "experiment_id": exp_id,
                    "llm_name": GENERATOR_NAME,
                    "llm_revision": GENERATOR_REVISION,
                    "sampling_seed": sampling_seed,
                    "temperature": float(config["generation"]["temperature"]),
                    "max_new_tokens": generator.max_steps,
                    "terminal_class": generated.terminal_class,
                    "success": generated.success,
                    "runtime_seconds": generated.runtime_seconds,
                    "status": generated.status,
                    "error_type": generated.error_type,
                }
            )
            for point in generated.checkpoints:
                checkpoints.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "checkpoint_id": point.checkpoint_id,
                        "trial_id": generated.trial_id,
                        "generated_token_index": point.generated_token_index,
                        "checkpoint_index": point.checkpoint_index,
                        "state_json": canonical_json(task.serialize_state(point.state)),
                        "objective_value": point.objective,
                        "optimality_gap": None,
                        "remaining_budget": point.remaining_budget,
                        "is_terminal": point.is_terminal,
                        "parse_status": "parsed",
                        "tensor_ref": None,
                    }
                )
                bits.append(list(point.state.bits))

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
                axis_names=("checkpoint", "bit"),
            )
        },
        trial_ids=[row["trial_id"] for row in checkpoints],
        checkpoint_ids=[row["checkpoint_id"] for row in checkpoints],
        generated_token_indices=[row["generated_token_index"] for row in checkpoints],
        observation_code_version=str(config["observation"]["code_version"]),
        observation_metadata={"source": "mock", "observation_kind": "external-only"},
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
    model = HammingDistanceStateModel(bit_count=int(config["task"]["bit_count"]))
    train_features = [row for row in features if row["instance_id"] in train_ids]
    model.fit(
        train_features,
        fit_split="train",
        split_hash=split.split_hash,
        instance_ids=train_ids,
    )
    transformed = model.transform(features)
    _write_json(paths.derived / "state_assignments.json", transformed)
    model.save(paths.artifacts / "state_model.json")
    train_trajectories = _trajectories_from_states(
        [row for row in transformed if row["instance_id"] in train_ids]
    )
    dynamics = MarkovTransitionModel(
        state_count=model.bit_count + 1,
        smoothing=float(config["dynamics"]["smoothing"]),
    )
    dynamics.fit(
        [row["states"] for row in train_trajectories],
        fit_split="train",
        split_hash=split.split_hash,
        instance_ids=train_ids,
    )
    dynamics.save(paths.artifacts / "dynamics_model.json")


def _evaluate(config: dict[str, Any], paths: PilotPaths) -> None:
    started_at = datetime.now(UTC)
    if int(config["evaluation"]["horizon"]) != 1:
        raise ValueError("Phase 2 evaluation implements only the one-step horizon")
    split = read_split(paths.derived / "splits.json")
    state_model = HammingDistanceStateModel.load(paths.artifacts / "state_model.json")
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
        raise ValueError("No held-out transitions are available for evaluation")
    validation = validate_dataset(paths.raw)
    if not validation.is_valid:
        raise ValueError(f"Dataset validation failed with {validation.error_count} errors")
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
        "generator_name": GENERATOR_NAME,
        "generator_revision": GENERATOR_REVISION,
        "dvc_revision": provenance["dvc_revision"],
    }

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
            },
        )
        return [metric_path, run_path, summary_path]

    record_run(
        tracking_uri=paths.tracking_uri,
        experiment_name=config["tracking"]["experiment_name"],
        parameters=config,
        metrics={**result.metrics, "evaluated_transitions": float(result.n_transitions)},
        tags=tags,
        artifacts=[
            paths.artifacts / "resolved_config.json",
            paths.derived / "splits.json",
            paths.artifacts / "state_model.json",
            paths.artifacts / "dynamics_model.json",
            paths.derived / "predictions.json",
            paths.artifacts / "provenance.json",
            paths.artifacts / "validation.json",
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
        "run_id": run["run_id"],
        "config_hash": summary["config_hash"],
        "task": f"{config['task']['name']}@{config['task']['version']}",
        "generator": f"{GENERATOR_NAME}@{GENERATOR_REVISION}",
        "instances": instances.num_rows,
        "trials": len(trials),
        "checkpoints": len(checkpoints),
        "train_instances": len(split.instance_ids("train")),
        "validation_instances": len(split.instance_ids("validation")),
        "test_instances": len(split.instance_ids("test")),
        "success_rate": sum(row["success"] for row in trials) / len(trials),
        "mean_final_hamming_distance": sum(distances) / len(distances),
        "one_step_nll": summary["one_step_nll"],
        "one_step_accuracy": summary["one_step_accuracy"],
        "success_brier_score": summary["success_brier_score"],
        "evaluated_transitions": summary["n_transitions"],
        "evaluated_instances": summary["n_instances"],
        "evaluated_trials": summary["n_trials"],
    }
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
    "collect",
    "prepare",
    "extract",
    "split",
    "fit",
    "evaluate",
    "report",
)


def run_stage(
    stage: str, config: dict[str, Any], *, root: Path | None = None, force: bool = False
) -> list[Path]:
    """Run one independent stage after checking prerequisites and output collisions."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown pilot stage: {stage}")
    paths = PilotPaths.from_config(config, root=root)
    outputs = _guard_outputs(paths, stage, force=force)
    if stage == "generate-instances":
        _generate_instances(config, paths)
    elif stage == "collect":
        _collect(config, paths)
    elif stage == "prepare":
        _prepare(config, paths)
    elif stage == "extract":
        _extract(paths)
    elif stage == "split":
        _split(config, paths)
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
    return {stage: run_stage(stage, config, root=root, force=force) for stage in STAGE_NAMES}
