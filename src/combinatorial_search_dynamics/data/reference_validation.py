"""Optional knapsack reference and checkpoint semantic validation."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa

from combinatorial_search_dynamics.data.parquet import read_parquet
from combinatorial_search_dynamics.data.references import reference_id
from combinatorial_search_dynamics.data.schemas import schema_issues
from combinatorial_search_dynamics.data.validation import ValidationIssue
from combinatorial_search_dynamics.identifiers import canonical_json
from combinatorial_search_dynamics.solvers.base import SolverStatus
from combinatorial_search_dynamics.solvers.ortools_knapsack import relative_maximization_gap
from combinatorial_search_dynamics.tasks.knapsack import KnapsackInstance, KnapsackTask


def _error(
    issues: list[ValidationIssue], code: str, artifact: str, related_id: str | None = None
) -> None:
    issues.append(ValidationIssue("error", code, code.replace("_", " "), artifact, related_id))


def _instances(table: pa.Table, issues: list[ValidationIssue]) -> dict[str, KnapsackInstance]:
    instances: dict[str, KnapsackInstance] = {}
    if not {"instance_id", "instance_json", "task_name"} <= set(table.column_names):
        return instances
    for row in table.to_pylist():
        if row["task_name"] != "knapsack":
            continue
        identifier = row["instance_id"]
        try:
            instance = KnapsackInstance.from_dict(json.loads(row["instance_json"]))
            instance.validate_structure()
            if instance.generation_seed != row["generation_seed"]:
                raise ValueError("generation seed mismatch")
            instances[identifier] = instance
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            _error(issues, "invalid_knapsack_instance", "instances", identifier)
    return instances


def validate_references(
    references: pa.Table,
    instances: pa.Table,
    issues: list[ValidationIssue],
) -> dict[str, dict[str, Any]]:
    """Validate optional references; return one row per known instance."""
    for problem in schema_issues(references, "reference_solutions"):
        _error(issues, problem.code, "reference_solutions")
    known = _instances(instances, issues)
    from combinatorial_search_dynamics.data.schemas import get_schema

    required = set(get_schema("reference_solutions").names)
    if not required <= set(references.column_names):
        return {}
    by_instance: dict[str, dict[str, Any]] = {}
    seen_keys: set[tuple[Any, ...]] = set()
    rows = references.to_pylist()
    reference_counts = Counter(row["reference_id"] for row in rows)
    for row in rows:
        identifier = row["instance_id"]
        if identifier not in known:
            _error(issues, "reference_foreign_key", "reference_solutions", identifier)
            continue
        instance = known[identifier]
        if row["task_name"] != instance.task_name or row["task_version"] != instance.task_version:
            _error(issues, "reference_task_mismatch", "reference_solutions", identifier)
        key = (
            identifier,
            row["solver_name"],
            row["solver_version"],
            row["solver_parameters_json"],
            row["solve_seed"],
        )
        if key in seen_keys:
            _error(issues, "reference_duplicate_key", "reference_solutions", identifier)
        seen_keys.add(key)
        if not row["reference_id"] or reference_counts[row["reference_id"]] > 1:
            _error(issues, "reference_id_duplicate", "reference_solutions", identifier)
        try:
            parameters = json.loads(row["solver_parameters_json"])
            if not isinstance(parameters, dict) or row["solver_parameters_json"] != canonical_json(
                parameters
            ):
                raise ValueError("noncanonical parameters")
            if parameters.get("random_seed") != row["solve_seed"]:
                _error(issues, "reference_seed_mismatch", "reference_solutions", identifier)
            expected_id = reference_id(
                identifier,
                solver_name=row["solver_name"],
                solver_version=row["solver_version"],
                parameters=parameters,
                solve_seed=row["solve_seed"],
            )
            if row["reference_id"] != expected_id:
                _error(issues, "reference_id_mismatch", "reference_solutions", identifier)
        except (TypeError, ValueError, json.JSONDecodeError):
            _error(issues, "reference_parameters_invalid", "reference_solutions", identifier)
            parameters = {}
        try:
            status = SolverStatus(row["solver_status"])
        except ValueError:
            _error(issues, "solver_status_invalid", "reference_solutions", identifier)
            continue
        proven = row["optimality_proven"]
        optimal = row["optimal_value"]
        feasible_value = row["best_feasible_value"]
        bound = row["best_bound"]
        for field in (
            "best_feasible_value",
            "best_bound",
            "optimal_value",
            "optimality_gap",
            "runtime_seconds",
        ):
            value = row[field]
            if value is not None and not math.isfinite(value):
                _error(issues, "solver_nonfinite_value", "reference_solutions", identifier)
        if proven != (status == SolverStatus.OPTIMAL):
            _error(issues, "optimality_status_mismatch", "reference_solutions", identifier)
        if (optimal is not None) != bool(proven):
            _error(issues, "unproven_optimal_value", "reference_solutions", identifier)
        if status == SolverStatus.FEASIBLE and optimal is not None:
            _error(issues, "feasible_has_optimal_value", "reference_solutions", identifier)
        if status == SolverStatus.OPTIMAL and (
            row["optimality_gap"] is None or abs(row["optimality_gap"]) > 1e-9
        ):
            _error(issues, "optimal_gap_nonzero", "reference_solutions", identifier)
        if status == SolverStatus.OPTIMAL and (
            feasible_value is None
            or bound is None
            or optimal is None
            or not math.isclose(optimal, feasible_value, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(bound, feasible_value, rel_tol=1e-9, abs_tol=1e-9)
        ):
            _error(issues, "optimal_value_mismatch", "reference_solutions", identifier)
        if feasible_value is not None and bound is not None and bound < feasible_value - 1e-9:
            _error(issues, "solver_bound_direction", "reference_solutions", identifier)
        if status in {
            SolverStatus.INFEASIBLE,
            SolverStatus.MODEL_INVALID,
            SolverStatus.UNKNOWN,
            SolverStatus.ERROR,
        } and (feasible_value is not None or row["solution_json"] is not None):
            _error(issues, "solver_status_solution_mismatch", "reference_solutions", identifier)
        if status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE} and (
            feasible_value is None or row["solution_json"] is None
        ):
            _error(issues, "solver_missing_solution", "reference_solutions", identifier)
        if row["timed_out"] and status in {
            SolverStatus.OPTIMAL,
            SolverStatus.INFEASIBLE,
            SolverStatus.MODEL_INVALID,
            SolverStatus.ERROR,
        }:
            _error(issues, "solver_timeout_status_mismatch", "reference_solutions", identifier)
        if row["runtime_seconds"] < 0:
            _error(issues, "solver_runtime_invalid", "reference_solutions", identifier)
        if feasible_value is not None and bound is not None:
            try:
                gap = relative_maximization_gap(
                    feasible_value,
                    bound,
                    epsilon=parameters["relative_gap_epsilon"],
                    optimality_proven=proven,
                )
                if row["optimality_gap"] is None or not math.isclose(
                    row["optimality_gap"], gap, rel_tol=1e-8, abs_tol=1e-9
                ):
                    _error(issues, "solver_gap_mismatch", "reference_solutions", identifier)
            except (KeyError, TypeError, ValueError):
                _error(issues, "reference_parameters_invalid", "reference_solutions", identifier)
        elif row["optimality_gap"] is not None:
            _error(issues, "solver_gap_without_solution", "reference_solutions", identifier)
        if row["solution_json"] is not None:
            try:
                vector = json.loads(row["solution_json"])
                if row["solution_json"] != canonical_json(vector):
                    raise ValueError("noncanonical solution")
                if (
                    not isinstance(vector, list)
                    or len(vector) != instance.item_count
                    or any(value not in (0, 1) or isinstance(value, bool) for value in vector)
                ):
                    raise ValueError("invalid solution vector")
                weight = sum(
                    weight
                    for weight, choose in zip(instance.weights, vector, strict=True)
                    if choose
                )
                value = sum(
                    value for value, choose in zip(instance.values, vector, strict=True) if choose
                )
                if weight > instance.capacity:
                    _error(issues, "solver_solution_infeasible", "reference_solutions", identifier)
                if feasible_value is None or not math.isclose(value, feasible_value, abs_tol=1e-9):
                    _error(issues, "solver_objective_mismatch", "reference_solutions", identifier)
            except (TypeError, ValueError, json.JSONDecodeError):
                _error(issues, "solver_solution_invalid", "reference_solutions", identifier)
        if (
            identifier in by_instance
            and by_instance[identifier]["reference_id"] != row["reference_id"]
        ):
            _error(issues, "reference_ambiguous_instance", "reference_solutions", identifier)
        else:
            by_instance[identifier] = row
    return by_instance


def validate_knapsack_dataset(
    directory: Path, tables: dict[str, pa.Table], issues: list[ValidationIssue]
) -> None:
    """Check reference, feasibility, action transitions, gap and trial success."""
    instances = tables.get("instances")
    if instances is None or "task_name" not in instances.column_names:
        return
    if "knapsack" not in set(instances.column("task_name").to_pylist()):
        return
    reference_path = directory / "reference_solutions.parquet"
    if not reference_path.is_file():
        _error(issues, "reference_missing", "reference_solutions")
        return
    try:
        references = read_parquet(reference_path, "reference_solutions")
    except (OSError, ValueError, pa.ArrowException):
        _error(issues, "reference_schema_invalid", "reference_solutions")
        return
    by_instance = validate_references(references, instances, issues)
    trials = tables.get("trials")
    checkpoints = tables.get("checkpoints")
    if trials is None or checkpoints is None:
        return
    trial_rows = {row["trial_id"]: row for row in trials.to_pylist()}
    task_instances = _instances(instances, issues)
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in checkpoints.to_pylist():
        by_trial.setdefault(row["trial_id"], []).append(row)
    for trial_id, points in by_trial.items():
        trial = trial_rows.get(trial_id)
        if trial is None or trial["instance_id"] not in task_instances:
            continue
        identifier = trial["instance_id"]
        instance = task_instances[identifier]
        legacy = "max_new_tokens" in trial
        budget_limit = int(trial["max_new_tokens"] if legacy else trial["budget_limit"])
        task = KnapsackTask.for_stored_instance(instance, max_steps=budget_limit)
        reference = by_instance.get(identifier)
        optimal = (
            reference["optimal_value"] if reference and reference["optimality_proven"] else None
        )
        epsilon = None
        if reference:
            try:
                epsilon = float(
                    json.loads(reference["solver_parameters_json"])["relative_gap_epsilon"]
                )
                if not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("invalid epsilon")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                _error(issues, "reference_parameters_invalid", "reference_solutions", identifier)
        previous = None
        for row in sorted(points, key=lambda point: point["checkpoint_index"]):
            try:
                payload = json.loads(row["state_json"])
                if isinstance(payload, dict) and isinstance(payload.get("selected"), list):
                    vector = payload["selected"]
                    if len(vector) == instance.item_count and all(
                        value in (0, 1) and not isinstance(value, bool) for value in vector
                    ):
                        weight = sum(
                            w
                            for w, selected in zip(instance.weights, vector, strict=True)
                            if selected
                        )
                        value = sum(
                            v
                            for v, selected in zip(instance.values, vector, strict=True)
                            if selected
                        )
                        if payload.get("total_weight") != weight:
                            _error(
                                issues,
                                "checkpoint_total_weight_mismatch",
                                "checkpoints",
                                row["checkpoint_id"],
                            )
                        if payload.get("total_value") != value:
                            _error(
                                issues,
                                "checkpoint_total_value_mismatch",
                                "checkpoints",
                                row["checkpoint_id"],
                            )
                        if weight > instance.capacity:
                            _error(
                                issues, "checkpoint_infeasible", "checkpoints", row["checkpoint_id"]
                            )
                state = task.deserialize_state(payload)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                _error(issues, "checkpoint_parse_failed", "checkpoints", row["checkpoint_id"])
                continue
            if (
                state.weights != instance.weights
                or state.values != instance.values
                or state.capacity != instance.capacity
            ):
                _error(issues, "checkpoint_instance_mismatch", "checkpoints", row["checkpoint_id"])
            expected_step = row["checkpoint_index"] if legacy else row["accepted_moves"]
            if state.step != expected_step:
                _error(issues, "checkpoint_step_mismatch", "checkpoints", row["checkpoint_id"])
            if not task.is_feasible(state):
                _error(issues, "checkpoint_infeasible", "checkpoints", row["checkpoint_id"])
            if row["objective_value"] is None or not math.isclose(
                row["objective_value"], task.objective(state), abs_tol=1e-9
            ):
                _error(issues, "checkpoint_objective_mismatch", "checkpoints", row["checkpoint_id"])
            expected_gap = (
                task.objective_gap(task.objective(state), optimal, epsilon=epsilon)
                if optimal is not None and epsilon is not None
                else None
            )
            actual_gap = row["optimality_gap"]
            if optimal is not None and epsilon is None:
                pass  # malformed reference parameters were reported above
            elif expected_gap is None and actual_gap is not None:
                _error(issues, "unproven_checkpoint_gap", "checkpoints", row["checkpoint_id"])
            elif expected_gap is not None and (
                actual_gap is None or not math.isclose(expected_gap, actual_gap, abs_tol=1e-9)
            ):
                _error(issues, "checkpoint_gap_mismatch", "checkpoints", row["checkpoint_id"])
            if previous is not None:
                changed = [
                    i
                    for i, (a, b) in enumerate(zip(previous.selected, state.selected, strict=True))
                    if a != b
                ]
                action_accepted = legacy or row["action_json"] is not None
                action_index = changed[0] if len(changed) == 1 else None
                if not legacy and row["action_json"] is not None:
                    try:
                        action_payload = json.loads(row["action_json"])
                        action_accepted = bool(action_payload["accepted"])
                        action_index = int(action_payload["bit_index"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        _error(
                            issues, "checkpoint_action_invalid", "checkpoints", row["checkpoint_id"]
                        )
                expected_changes = 1 if action_accepted else 0
                if len(changed) != expected_changes:
                    _error(
                        issues, "checkpoint_action_mismatch", "checkpoints", row["checkpoint_id"]
                    )
                elif action_accepted and action_index is not None:
                    try:
                        applied = task.apply_action(previous, action_index)
                        if applied != state:
                            _error(
                                issues,
                                "checkpoint_action_mismatch",
                                "checkpoints",
                                row["checkpoint_id"],
                            )
                    except ValueError:
                        _error(
                            issues,
                            "checkpoint_action_mismatch",
                            "checkpoints",
                            row["checkpoint_id"],
                        )
            previous = state
        if trial["success"] and (
            optimal is None or previous is None or not task.is_success(previous, optimal)
        ):
            _error(issues, "trial_success_reference_mismatch", "trials", trial_id)
