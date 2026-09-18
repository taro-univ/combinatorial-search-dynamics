"""Dataset-wide structural and relational validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from llm_search_dynamics.data.schemas import (
    TABLE_NAMES,
    UnsupportedSchemaVersionError,
    get_schema,
    schema_issues,
)
from llm_search_dynamics.data.zarr import (
    ZarrValidationError,
    read_observation_store,
)

ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_TRIAL_STATUSES = frozenset({"pending", "running", "completed", "failed", "interrupted"})
SUCCESS_TERMINAL_CLASSES = frozenset({"success", "optimal", "feasible"})


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    table_or_artifact: str
    related_id: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)


def _issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    artifact: str,
    related_id: str | None = None,
) -> None:
    issues.append(ValidationIssue("error", code, message, artifact, related_id))


def _values(table: pa.Table, column: str) -> list[object]:
    if column not in table.column_names:
        return []
    return table.column(column).to_pylist()


def _compatible_columns(table: pa.Table, table_name: str, columns: set[str]) -> bool:
    if not columns <= set(table.column_names):
        return False
    expected = get_schema(table_name)
    return all(table.schema.field(name).type == expected.field(name).type for name in columns)


def _validate_primary_key(
    table: pa.Table,
    table_name: str,
    columns: tuple[str, ...],
    issues: list[ValidationIssue],
) -> None:
    if any(column not in table.column_names for column in columns):
        return
    keys = list(zip(*(_values(table, column) for column in columns), strict=True))
    for row_index, key in enumerate(keys):
        if any(value is None for value in key):
            _issue(
                issues,
                "primary_key_null",
                f"Primary key contains null at row {row_index}",
                table_name,
            )
    for key, count in Counter(
        key for key in keys if all(value is not None for value in key)
    ).items():
        if count > 1:
            related = str(key[0]) if key else None
            _issue(
                issues, "primary_key_duplicate", "Primary key is duplicated", table_name, related
            )


def _load_tables(directory: Path, issues: list[ValidationIssue]) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}
    for table_name in TABLE_NAMES:
        path = directory / f"{table_name}.parquet"
        if not path.is_file():
            _issue(issues, "missing_artifact", f"Missing {path.name}", table_name)
            continue
        try:
            table = pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            _issue(
                issues,
                "parquet_read",
                f"Could not read {path.name}: {type(exc).__name__}",
                table_name,
            )
            continue
        tables[table_name] = table
        for problem in schema_issues(table, table_name):
            _issue(issues, problem.code, problem.message, table_name)
    return tables


def _validate_keys(tables: dict[str, pa.Table], issues: list[ValidationIssue]) -> None:
    key_columns = {
        "instances": ("instance_id",),
        "trials": ("trial_id",),
        "checkpoints": ("checkpoint_id",),
    }
    for table_name, columns in key_columns.items():
        if table_name in tables and _compatible_columns(
            tables[table_name], table_name, set(columns)
        ):
            _validate_primary_key(tables[table_name], table_name, columns, issues)


def _validate_relations(tables: dict[str, pa.Table], issues: list[ValidationIssue]) -> None:
    instances = tables.get("instances")
    trials = tables.get("trials")
    checkpoints = tables.get("checkpoints")
    if (
        instances is not None
        and trials is not None
        and _compatible_columns(instances, "instances", {"instance_id"})
        and _compatible_columns(trials, "trials", {"instance_id"})
    ):
        known_instances = set(_values(instances, "instance_id"))
        for value in _values(trials, "instance_id"):
            if value is not None and value not in known_instances:
                _issue(
                    issues,
                    "foreign_key",
                    "trials.instance_id has no matching instances row",
                    "trials",
                    str(value),
                )
    if (
        trials is not None
        and checkpoints is not None
        and _compatible_columns(trials, "trials", {"trial_id"})
        and _compatible_columns(checkpoints, "checkpoints", {"trial_id"})
    ):
        known_trials = set(_values(trials, "trial_id"))
        for value in _values(checkpoints, "trial_id"):
            if value is not None and value not in known_trials:
                _issue(
                    issues,
                    "foreign_key",
                    "checkpoints.trial_id has no matching trials row",
                    "checkpoints",
                    str(value),
                )


def _validate_checkpoint_order(table: pa.Table, issues: list[ValidationIssue]) -> None:
    required = {"trial_id", "checkpoint_index", "generated_token_index"}
    if not _compatible_columns(table, "checkpoints", required):
        return
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    rows = zip(
        _values(table, "trial_id"),
        _values(table, "checkpoint_index"),
        _values(table, "generated_token_index"),
        strict=True,
    )
    for trial, checkpoint_index, token_index in rows:
        if trial is None or checkpoint_index is None or token_index is None:
            continue
        grouped[str(trial)].append((int(checkpoint_index), int(token_index)))

    for trial, positions in grouped.items():
        checkpoint_indexes = [position[0] for position in positions]
        if len(checkpoint_indexes) != len(set(checkpoint_indexes)):
            _issue(
                issues,
                "checkpoint_index_duplicate",
                "checkpoint_index is duplicated within a trial",
                "checkpoints",
                trial,
            )
        ordered_tokens = [token for _, token in sorted(positions)]
        if any(current < previous for previous, current in pairwise(ordered_tokens)):
            _issue(
                issues,
                "checkpoint_order",
                "generated_token_index decreases as checkpoint_index increases",
                "checkpoints",
                trial,
            )


def _validate_enums_and_status(tables: dict[str, pa.Table], issues: list[ValidationIssue]) -> None:
    metrics = tables.get("metrics")
    if metrics is not None and _compatible_columns(metrics, "metrics", {"split"}):
        for split in _values(metrics, "split"):
            if split is not None and split not in ALLOWED_SPLITS:
                _issue(issues, "invalid_split", "metrics.split is not allowed", "metrics")

    trials = tables.get("trials")
    if trials is None:
        return
    required = {"trial_id", "status", "success", "terminal_class", "error_type"}
    if not _compatible_columns(trials, "trials", required):
        return
    rows = zip(
        _values(trials, "trial_id"),
        _values(trials, "status"),
        _values(trials, "success"),
        _values(trials, "terminal_class"),
        _values(trials, "error_type"),
        strict=True,
    )
    for trial, status, success, terminal_class, error_type in rows:
        related = str(trial) if trial is not None else None
        if status is not None and status not in ALLOWED_TRIAL_STATUSES:
            _issue(issues, "invalid_status", "trials.status is not allowed", "trials", related)
        contradiction = success is True and (status != "completed" or error_type is not None)
        contradiction |= success is False and terminal_class in SUCCESS_TERMINAL_CLASSES
        if contradiction:
            _issue(
                issues,
                "success_terminal_contradiction",
                "success conflicts with status, error_type, or terminal_class",
                "trials",
                related,
            )


def _validate_zarr(
    directory: Path,
    checkpoints: pa.Table | None,
    issues: list[ValidationIssue],
) -> None:
    store_path = directory / "observations.zarr"
    if not store_path.is_dir():
        _issue(issues, "missing_artifact", "Missing observations.zarr", "observations.zarr")
        return
    try:
        store = read_observation_store(store_path)
    except (OSError, UnsupportedSchemaVersionError, ZarrValidationError) as exc:
        _issue(
            issues,
            "zarr_structure",
            f"Invalid observations.zarr: {type(exc).__name__}",
            "observations.zarr",
        )
        return
    if checkpoints is None:
        return
    required = {"checkpoint_id", "trial_id", "generated_token_index"}
    if not _compatible_columns(checkpoints, "checkpoints", required):
        return

    parquet_index = {
        str(checkpoint): (str(trial), int(token))
        for checkpoint, trial, token in zip(
            _values(checkpoints, "checkpoint_id"),
            _values(checkpoints, "trial_id"),
            _values(checkpoints, "generated_token_index"),
            strict=True,
        )
        if checkpoint is not None and trial is not None and token is not None
    }
    if len(store.checkpoint_ids) != len(set(store.checkpoint_ids)):
        _issue(
            issues,
            "zarr_checkpoint_duplicate",
            "Zarr checkpoint_id index contains duplicates",
            "observations.zarr",
        )
    zarr_ids = set(store.checkpoint_ids)
    for checkpoint in sorted(set(parquet_index) - zarr_ids):
        _issue(
            issues,
            "zarr_checkpoint_missing",
            "Parquet checkpoint has no Zarr index entry",
            "observations.zarr",
            checkpoint,
        )
    for checkpoint, trial, token in zip(
        store.checkpoint_ids,
        store.trial_ids,
        store.generated_token_indices.tolist(),
        strict=True,
    ):
        if checkpoint not in parquet_index:
            _issue(
                issues,
                "zarr_checkpoint_unknown",
                "Zarr checkpoint_id has no Parquet checkpoint row",
                "observations.zarr",
                checkpoint,
            )
        elif parquet_index[checkpoint] != (trial, int(token)):
            _issue(
                issues,
                "zarr_index_mismatch",
                "Zarr trial_id or token index differs from Parquet",
                "observations.zarr",
                checkpoint,
            )


def validate_dataset(data_dir: Path, *, allow_empty: bool = False) -> ValidationResult:
    """Validate all Phase 1 artifacts while collecting every detectable issue."""
    directory = Path(data_dir).resolve()
    expected_paths = [directory / f"{name}.parquet" for name in TABLE_NAMES]
    expected_paths.append(directory / "observations.zarr")
    has_data = directory.is_dir() and any(path.exists() for path in expected_paths)
    incomplete_paths = (
        [
            path
            for path in directory.iterdir()
            if path.name.endswith((".tmp", ".partial", ".backup"))
        ]
        if directory.is_dir()
        else []
    )
    if not has_data and not incomplete_paths:
        if allow_empty:
            return ValidationResult(())
        return ValidationResult(
            (
                ValidationIssue(
                    "error",
                    "empty_dataset",
                    "No Phase 1 data artifacts were found",
                    str(directory),
                ),
            )
        )

    issues: list[ValidationIssue] = []
    for path in incomplete_paths:
        _issue(
            issues,
            "incomplete_artifact",
            "Temporary or incomplete artifact remains in the dataset directory",
            path.name,
        )

    tables = _load_tables(directory, issues)
    _validate_keys(tables, issues)
    _validate_relations(tables, issues)
    if "checkpoints" in tables:
        _validate_checkpoint_order(tables["checkpoints"], issues)
    _validate_enums_and_status(tables, issues)
    _validate_zarr(directory, tables.get("checkpoints"), issues)
    if "instances" in tables:
        from llm_search_dynamics.data.reference_validation import validate_knapsack_dataset

        validate_knapsack_dataset(directory, tables, issues)
    return ValidationResult(tuple(issues))
