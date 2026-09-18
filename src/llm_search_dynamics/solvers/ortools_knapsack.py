"""Single-worker CP-SAT reference solver for 0-1 knapsack."""

from __future__ import annotations

import math
import time

import ortools
from ortools.sat.python import cp_model

from llm_search_dynamics.solvers.base import (
    SolverParameters,
    SolverResult,
    SolverStatus,
)
from llm_search_dynamics.tasks.knapsack import KnapsackInstance, KnapsackTask


class SolverValidationError(ValueError):
    """A returned solver solution failed independent Task verification."""


def relative_maximization_gap(
    best_feasible: float | None,
    best_bound: float | None,
    *,
    epsilon: float,
    optimality_proven: bool = False,
) -> float | None:
    """Return max(0, bound - feasible) / max(abs(feasible), epsilon)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if best_feasible is None or best_bound is None:
        return None
    if optimality_proven:
        return 0.0
    return max(0.0, best_bound - best_feasible) / max(abs(best_feasible), epsilon)


def mapped_status(status: int) -> SolverStatus:
    """Translate CP-SAT statuses without exposing OR-Tools values to callers."""
    mapping = {
        cp_model.OPTIMAL: SolverStatus.OPTIMAL,
        cp_model.FEASIBLE: SolverStatus.FEASIBLE,
        cp_model.INFEASIBLE: SolverStatus.INFEASIBLE,
        cp_model.MODEL_INVALID: SolverStatus.MODEL_INVALID,
        cp_model.UNKNOWN: SolverStatus.UNKNOWN,
    }
    try:
        return mapping[status]
    except KeyError as exc:
        raise ValueError(f"Unrecognized CP-SAT solver status: {status}") from exc


def limit_reached(status: SolverStatus, *, runtime: float, time_limit: float) -> bool:
    """Classify a non-optimal time-limit result using an explicit 95% threshold."""
    return status in {SolverStatus.FEASIBLE, SolverStatus.UNKNOWN} and runtime >= 0.95 * time_limit


class OrtoolsKnapsackSolver:
    name = "ortools_cp_sat"
    version = ortools.__version__

    def __init__(self, *, task: KnapsackTask, parameters: SolverParameters) -> None:
        self.task = task
        self.parameters = parameters

    def solve(self, instance: KnapsackInstance, *, instance_id: str) -> SolverResult:
        self.task.validate_instance(instance)
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        model = cp_model.CpModel()
        variables = [model.new_bool_var(f"item_{index}") for index in range(instance.item_count)]
        model.add(
            sum(
                weight * variable
                for weight, variable in zip(instance.weights, variables, strict=True)
            )
            <= instance.capacity
        )
        model.maximize(
            sum(
                value * variable for value, variable in zip(instance.values, variables, strict=True)
            )
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.parameters.max_time_seconds
        solver.parameters.num_search_workers = self.parameters.num_search_workers
        solver.parameters.random_seed = self.parameters.random_seed
        solver.parameters.log_search_progress = self.parameters.log_search_progress
        started = time.perf_counter()
        status = mapped_status(solver.solve(model))
        runtime = time.perf_counter() - started
        feasible = status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
        solution = (
            tuple(int(solver.value(variable)) for variable in variables) if feasible else None
        )
        best_feasible = float(solver.objective_value) if feasible else None
        best_bound = float(solver.best_objective_bound)
        if not math.isfinite(best_bound):
            best_bound = None
        if feasible:
            state = self.task.initial_state(instance)
            try:
                for index, selected in enumerate(solution):
                    if selected:
                        state = self.task.apply_action(state, index)
            except ValueError as exc:
                raise SolverValidationError("CP-SAT returned an infeasible solution") from exc
            if not self.task.is_feasible(state):
                raise SolverValidationError("CP-SAT returned an infeasible solution")
            if not math.isclose(
                self.task.objective(state), best_feasible, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise SolverValidationError("CP-SAT objective differs from Task recomputation")
        proven = status == SolverStatus.OPTIMAL
        return SolverResult(
            instance_id=instance_id,
            solver_name=self.name,
            solver_version=self.version,
            status=status,
            best_feasible_value=best_feasible,
            best_bound=best_bound,
            optimal_value=best_feasible if proven else None,
            optimality_gap=relative_maximization_gap(
                best_feasible,
                best_bound,
                epsilon=self.parameters.relative_gap_epsilon,
                optimality_proven=proven,
            ),
            optimality_proven=proven,
            timed_out=limit_reached(
                status, runtime=runtime, time_limit=self.parameters.max_time_seconds
            ),
            runtime_seconds=runtime,
            solution=solution,
            parameters=self.parameters,
        )
