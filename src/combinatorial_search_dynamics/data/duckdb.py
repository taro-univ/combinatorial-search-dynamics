"""In-memory DuckDB views over canonical Parquet files."""

from __future__ import annotations

from pathlib import Path

import duckdb

from combinatorial_search_dynamics.data.parquet import read_parquet
from combinatorial_search_dynamics.data.schemas import TABLE_NAMES


class MissingParquetError(FileNotFoundError):
    """Raised when a required logical table file is absent."""


def create_analysis_views(
    data_dir: Path,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Attach validated Parquet files as views and create analysis_checkpoints.

    The returned connection is in-memory by default. Parquet remains the source
    of truth: no rows are copied into persistent DuckDB tables.
    """
    directory = Path(data_dir).resolve()
    paths = {name: directory / f"{name}.parquet" for name in TABLE_NAMES}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise MissingParquetError(f"Missing required Parquet files: {', '.join(missing)}")

    validated = {name: read_parquet(path, name) for name, path in paths.items()}

    conn = connection or duckdb.connect(":memory:")
    for name, path in paths.items():
        conn.read_parquet(str(path)).create_view(name, replace=True)

    v2 = "search_method_name" in validated["trials"].column_names
    trial_fields = (
        "t.search_method_name, t.search_method_revision, t.search_seed, "
        "t.budget_type, t.budget_limit, t.search_parameters_json, t.initial_state_json"
        if v2
        else "t.llm_name, t.llm_revision, t.sampling_seed, t.temperature, t.max_new_tokens"
    )
    checkpoint_fields = (
        "c.budget_used, c.checkpoint_index, c.decision_step, c.accepted_moves, "
        "c.rejected_moves, c.action_json"
        if v2
        else "c.generated_token_index, c.checkpoint_index, c.parse_status"
    )
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW analysis_checkpoints AS
        SELECT
            i.schema_version AS instance_schema_version,
            i.instance_id,
            i.task_name,
            i.task_version,
            i.problem_size,
            i.difficulty_value,
            i.generation_seed,
            i.instance_json,
            i.created_at,
            t.schema_version AS trial_schema_version,
            t.trial_id,
            t.experiment_id,
            {trial_fields},
            t.terminal_class,
            t.success,
            t.runtime_seconds,
            t.status,
            t.error_type,
            c.schema_version AS checkpoint_schema_version,
            c.checkpoint_id,
            {checkpoint_fields},
            c.state_json,
            c.objective_value,
            c.optimality_gap,
            c.remaining_budget,
            c.is_terminal,
            c.tensor_ref
        FROM instances AS i
        INNER JOIN trials AS t ON i.instance_id = t.instance_id
        INNER JOIN checkpoints AS c ON t.trial_id = c.trial_id
        """
    )
    return conn
