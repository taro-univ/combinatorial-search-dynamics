"""Stable reference IDs and canonical rows for the optional solver table."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from llm_search_dynamics.data.schemas import SCHEMA_VERSION
from llm_search_dynamics.identifiers import artifact_id, canonical_json
from llm_search_dynamics.solvers.base import SolverResult


def reference_id(
    instance_id: str,
    *,
    solver_name: str,
    solver_version: str,
    parameters: dict[str, Any],
    solve_seed: int,
) -> str:
    """Identify a reference by instance and complete solver conditions."""
    return artifact_id(
        {
            "kind": "reference_solution",
            "instance_id": instance_id,
            "solver_name": solver_name,
            "solver_version": solver_version,
            "parameters": parameters,
            "solve_seed": solve_seed,
        }
    )


def reference_row(
    result: SolverResult,
    *,
    task_name: str,
    task_version: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    parameters = result.parameters.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_id": reference_id(
            result.instance_id,
            solver_name=result.solver_name,
            solver_version=result.solver_version,
            parameters=parameters,
            solve_seed=result.parameters.random_seed,
        ),
        "instance_id": result.instance_id,
        "task_name": task_name,
        "task_version": task_version,
        "solver_name": result.solver_name,
        "solver_version": result.solver_version,
        "solver_status": result.status.value,
        "best_feasible_value": result.best_feasible_value,
        "best_bound": result.best_bound,
        "optimal_value": result.optimal_value,
        "optimality_gap": result.optimality_gap,
        "optimality_proven": result.optimality_proven,
        "timed_out": result.timed_out,
        "runtime_seconds": result.runtime_seconds,
        "solution_json": canonical_json(list(result.solution))
        if result.solution is not None
        else None,
        "solver_parameters_json": canonical_json(parameters),
        "solve_seed": result.parameters.random_seed,
        "error_type": result.error_type,
        "created_at": created_at or datetime.now(UTC),
    }
