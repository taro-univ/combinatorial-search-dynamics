"""Canonical PyArrow schemas and strict compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

SCHEMA_VERSION = "1"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when persisted data uses an unsupported schema version."""


class SchemaValidationError(ValueError):
    """Raised when a table does not match its canonical schema."""


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    message: str


def _schema(table_name: str, fields: list[pa.Field]) -> pa.Schema:
    metadata = {
        b"logical_table": table_name.encode("utf-8"),
        b"schema_version": SCHEMA_VERSION.encode("ascii"),
    }
    return pa.schema([pa.field("schema_version", pa.string(), nullable=False), *fields], metadata)


_SCHEMAS = {
    "instances": _schema(
        "instances",
        [
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("task_name", pa.string(), nullable=False),
            pa.field("task_version", pa.string(), nullable=False),
            pa.field("problem_size", pa.int64(), nullable=False),
            pa.field("difficulty_value", pa.float64(), nullable=True),
            pa.field("generation_seed", pa.int64(), nullable=False),
            pa.field("instance_json", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ],
    ),
    "trials": _schema(
        "trials",
        [
            pa.field("trial_id", pa.string(), nullable=False),
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("llm_name", pa.string(), nullable=False),
            pa.field("llm_revision", pa.string(), nullable=True),
            pa.field("sampling_seed", pa.int64(), nullable=False),
            pa.field("temperature", pa.float64(), nullable=False),
            pa.field("max_new_tokens", pa.int64(), nullable=False),
            pa.field("terminal_class", pa.string(), nullable=False),
            pa.field("success", pa.bool_(), nullable=False),
            pa.field("runtime_seconds", pa.float64(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("error_type", pa.string(), nullable=True),
        ],
    ),
    "checkpoints": _schema(
        "checkpoints",
        [
            pa.field("checkpoint_id", pa.string(), nullable=False),
            pa.field("trial_id", pa.string(), nullable=False),
            pa.field("generated_token_index", pa.int64(), nullable=False),
            pa.field("checkpoint_index", pa.int64(), nullable=False),
            pa.field("state_json", pa.string(), nullable=True),
            pa.field("objective_value", pa.float64(), nullable=True),
            pa.field("optimality_gap", pa.float64(), nullable=True),
            pa.field("remaining_budget", pa.int64(), nullable=False),
            pa.field("is_terminal", pa.bool_(), nullable=False),
            pa.field("parse_status", pa.string(), nullable=False),
            pa.field("tensor_ref", pa.string(), nullable=True),
        ],
    ),
    "metrics": _schema(
        "metrics",
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("fold", pa.int64(), nullable=True),
            pa.field("horizon", pa.int64(), nullable=True),
            pa.field("metric_name", pa.string(), nullable=False),
            pa.field("metric_value", pa.float64(), nullable=False),
            pa.field("n_units", pa.int64(), nullable=False),
        ],
    ),
    "reference_solutions": _schema(
        "reference_solutions",
        [
            pa.field("reference_id", pa.string(), nullable=False),
            pa.field("instance_id", pa.string(), nullable=False),
            pa.field("task_name", pa.string(), nullable=False),
            pa.field("task_version", pa.string(), nullable=False),
            pa.field("solver_name", pa.string(), nullable=False),
            pa.field("solver_version", pa.string(), nullable=False),
            pa.field("solver_status", pa.string(), nullable=False),
            pa.field("best_feasible_value", pa.float64(), nullable=True),
            pa.field("best_bound", pa.float64(), nullable=True),
            pa.field("optimal_value", pa.float64(), nullable=True),
            pa.field("optimality_gap", pa.float64(), nullable=True),
            pa.field("optimality_proven", pa.bool_(), nullable=False),
            pa.field("timed_out", pa.bool_(), nullable=False),
            pa.field("runtime_seconds", pa.float64(), nullable=False),
            pa.field("solution_json", pa.string(), nullable=True),
            pa.field("solver_parameters_json", pa.string(), nullable=False),
            pa.field("solve_seed", pa.int64(), nullable=False),
            pa.field("error_type", pa.string(), nullable=True),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ],
    ),
}

TABLE_NAMES = ("instances", "trials", "checkpoints", "metrics")
OPTIONAL_TABLE_NAMES = ("reference_solutions",)


def get_schema(table_name: str) -> pa.Schema:
    """Return the canonical schema or fail clearly for an unknown logical table."""
    try:
        return _SCHEMAS[table_name]
    except KeyError as exc:
        raise KeyError(f"Unknown logical table: {table_name}") from exc


def empty_table(table_name: str) -> pa.Table:
    """Create an empty table that retains the complete canonical schema."""
    schema = get_schema(table_name)
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def schema_issues(table: pa.Table, table_name: str) -> list[SchemaIssue]:
    """Return all detectable structural and version issues for an Arrow table."""
    expected = get_schema(table_name)
    expected_names = expected.names
    actual_names = table.schema.names
    missing = [name for name in expected_names if name not in actual_names]
    extra = [name for name in actual_names if name not in expected_names]
    issues: list[SchemaIssue] = []

    if missing:
        issues.append(SchemaIssue("missing_columns", f"Missing columns: {', '.join(missing)}"))
    if extra:
        issues.append(SchemaIssue("extra_columns", f"Unexpected columns: {', '.join(extra)}"))
    if not missing and not extra and actual_names != expected_names:
        issues.append(
            SchemaIssue("column_order", "Column order does not match the canonical schema")
        )

    for name in set(expected_names) & set(actual_names):
        expected_field = expected.field(name)
        actual_field = table.schema.field(name)
        if actual_field.type != expected_field.type:
            issues.append(
                SchemaIssue(
                    "column_type",
                    f"Column {name} has type {actual_field.type}; expected {expected_field.type}",
                )
            )
        if actual_field.nullable != expected_field.nullable:
            issues.append(
                SchemaIssue(
                    "column_nullability",
                    f"Column {name} nullable metadata does not match the canonical schema",
                )
            )
        if not expected_field.nullable and table.column(name).null_count:
            issues.append(SchemaIssue("null_value", f"Non-null column {name} contains nulls"))

    metadata = table.schema.metadata or {}
    stored_table = metadata.get(b"logical_table", b"").decode("utf-8", errors="replace")
    stored_version = metadata.get(b"schema_version", b"").decode("ascii", errors="replace")
    if stored_table != table_name:
        issues.append(
            SchemaIssue("table_metadata", "Logical table metadata is missing or incorrect")
        )
    if stored_version != SCHEMA_VERSION:
        issues.append(
            SchemaIssue(
                "schema_version",
                f"Unsupported schema metadata version {stored_version or '<missing>'}",
            )
        )

    if "schema_version" in actual_names:
        versions = set(table.column("schema_version").to_pylist())
        if versions - {SCHEMA_VERSION}:
            issues.append(SchemaIssue("schema_version", "schema_version column is unsupported"))
    return issues


def validate_table_schema(table: pa.Table, table_name: str) -> None:
    """Raise when a table differs from its exact canonical schema."""
    issues = schema_issues(table, table_name)
    version_issues = [issue for issue in issues if issue.code == "schema_version"]
    if version_issues:
        raise UnsupportedSchemaVersionError("; ".join(issue.message for issue in version_issues))
    if issues:
        raise SchemaValidationError("; ".join(issue.message for issue in issues))
