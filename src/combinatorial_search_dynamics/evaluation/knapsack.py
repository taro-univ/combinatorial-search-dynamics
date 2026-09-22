"""Knapsack pilot summaries with explicit trial, instance and checkpoint units."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from combinatorial_search_dynamics.tasks.knapsack import KnapsackInstance, KnapsackTask


@dataclass(frozen=True)
class ProblemMetric:
    value: float
    n_units: int


@dataclass(frozen=True)
class KnapsackEvaluation:
    metrics: dict[str, ProblemMetric]
    report: dict[str, Any]


def evaluate_knapsack(
    *,
    task: KnapsackTask,
    instances: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    references: list[dict[str, Any]],
    test_instance_ids: set[str],
) -> KnapsackEvaluation:
    """Aggregate test outcomes without dropping timeout or unproven instances."""
    reference_by_instance = {row["instance_id"]: row for row in references}
    instance_by_id = {
        row["instance_id"]: KnapsackInstance.from_dict(json.loads(row["instance_json"]))
        for row in instances
    }
    trial_points: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        trial_points[checkpoint["trial_id"]].append(checkpoint)
    for points in trial_points.values():
        points.sort(key=lambda row: row["checkpoint_index"])
    test_trials = [row for row in trials if row["instance_id"] in test_instance_ids]
    if not test_trials:
        raise ValueError("No knapsack test trials are available")
    final_values: list[float] = []
    absolute_gaps: list[float] = []
    relative_gaps: list[float] = []
    best_values: list[float] = []
    reached_budgets: list[int] = []
    feasible_count = 0
    checkpoint_count = 0
    success_by_instance: dict[str, bool] = {identifier: False for identifier in test_instance_ids}
    for trial in test_trials:
        identifier = trial["instance_id"]
        points = trial_points[trial["trial_id"]]
        if not points:
            raise ValueError("Knapsack trial has no checkpoints")
        instance = instance_by_id[identifier]
        task.validate_instance(instance)
        objectives: list[float] = []
        for point in points:
            state = task.deserialize_state(json.loads(point["state_json"]))
            feasible_count += int(task.is_feasible(state))
            checkpoint_count += 1
            objectives.append(task.objective(state))
        final = objectives[-1]
        final_values.append(final)
        best_values.append(max(objectives))
        success_by_instance[identifier] |= bool(trial["success"])
        reference = reference_by_instance[identifier]
        if reference["optimality_proven"]:
            optimal = float(reference["optimal_value"])
            epsilon = float(json.loads(reference["solver_parameters_json"])["relative_gap_epsilon"])
            absolute_gaps.append(max(0.0, optimal - final))
            relative_gaps.append(task.objective_gap(final, optimal, epsilon=epsilon))
            for point, value in zip(points, objectives, strict=True):
                if value >= optimal - 1e-9:
                    reached_budgets.append(int(point["budget_used"]))
                    break
    test_references = [
        reference_by_instance[identifier] for identifier in sorted(test_instance_ids)
    ]
    metrics = {
        "knapsack_success_rate_trial": ProblemMetric(
            sum(bool(row["success"]) for row in test_trials) / len(test_trials), len(test_trials)
        ),
        "knapsack_success_rate_instance": ProblemMetric(
            sum(success_by_instance.values()) / len(success_by_instance), len(success_by_instance)
        ),
        "knapsack_final_value_mean": ProblemMetric(
            sum(final_values) / len(final_values), len(final_values)
        ),
        "knapsack_best_so_far_value_mean": ProblemMetric(
            sum(best_values) / len(best_values), len(best_values)
        ),
        "knapsack_feasible_checkpoint_rate": ProblemMetric(
            feasible_count / checkpoint_count, checkpoint_count
        ),
        "solver_timeout_rate_test": ProblemMetric(
            sum(bool(row["timed_out"]) for row in test_references) / len(test_references),
            len(test_references),
        ),
        "solver_optimality_proven_rate_test": ProblemMetric(
            sum(bool(row["optimality_proven"]) for row in test_references) / len(test_references),
            len(test_references),
        ),
    }
    if absolute_gaps:
        metrics["knapsack_final_absolute_gap_mean"] = ProblemMetric(
            sum(absolute_gaps) / len(absolute_gaps), len(absolute_gaps)
        )
        metrics["knapsack_final_relative_gap_mean"] = ProblemMetric(
            sum(relative_gaps) / len(relative_gaps), len(relative_gaps)
        )
    if reached_budgets:
        metrics["knapsack_optimal_reached_budget_mean"] = ProblemMetric(
            sum(reached_budgets) / len(reached_budgets), len(reached_budgets)
        )
    statuses = dict(sorted(Counter(row["solver_status"] for row in references).items()))
    report = {
        "solver_status_counts": statuses,
        "solver_timeout_count": sum(bool(row["timed_out"]) for row in references),
        "solver_optimality_proven_rate": sum(bool(row["optimality_proven"]) for row in references)
        / len(references),
        "solver_runtime_mean_seconds": sum(row["runtime_seconds"] for row in references)
        / len(references),
        "test_trial_count": len(test_trials),
        "test_instance_count": len(test_instance_ids),
    }
    return KnapsackEvaluation(metrics=metrics, report=report)
