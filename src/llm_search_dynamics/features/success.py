"""Leakage-safe B2, local-landscape, history, and instance features."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from llm_search_dynamics.tasks.knapsack import KnapsackInstance

B2_FEATURES = ("remaining_budget_fraction", "relative_objective_gap")
M2_FEATURES = (
    "M2_distance",
    "M2_improving_fraction",
    "M2_best_gain",
    "M2_local_entropy",
)
M3_FEATURES = (
    "M3_progress_rate",
    "M3_stagnation",
    "M3_revisit",
    "M3_best_update_rate",
    "M3_visit_entropy",
)


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _coefficient_of_variation(values: tuple[int, ...] | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    return float(np.std(array) / abs(mean)) if abs(mean) > 1e-12 else 0.0


def neighborhood_features(
    instance: KnapsackInstance,
    selected: tuple[int, ...],
    *,
    objective_scale: float,
    epsilon: float,
) -> dict[str, float]:
    """Measure every one-bit neighbor without consuming online search budget."""
    weight = sum(w for w, bit in zip(instance.weights, selected, strict=True) if bit)
    value = sum(v for v, bit in zip(instance.values, selected, strict=True) if bit)
    counts = {"improving": 0, "equal": 0, "worsening": 0, "infeasible": 0}
    gains: list[float] = []
    feasible = 0
    for index, bit in enumerate(selected):
        candidate_weight = weight + (-instance.weights[index] if bit else instance.weights[index])
        if candidate_weight > instance.capacity:
            counts["infeasible"] += 1
            continue
        feasible += 1
        candidate_value = value + (-instance.values[index] if bit else instance.values[index])
        gain = float(candidate_value - value)
        gains.append(gain)
        if gain > epsilon:
            counts["improving"] += 1
        elif gain < -epsilon:
            counts["worsening"] += 1
        else:
            counts["equal"] += 1
    total = instance.item_count
    scale = max(abs(objective_scale), epsilon)
    return {
        "M2_improving_fraction": counts["improving"] / max(feasible, 1),
        "M2_best_gain": max([0.0, *gains]) / scale,
        "M2_local_entropy": _entropy([count / total for count in counts.values()]),
    }


def nearest_optimal_distance(
    instance: KnapsackInstance,
    selected: tuple[int, ...],
    optimal_value: float,
    *,
    solve_seed: int,
) -> float:
    """Return normalized Hamming distance to the closest certified optimal solution."""
    target = round(optimal_value)
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"x_{index}") for index in range(instance.item_count)]
    model.add(
        sum(w * x for w, x in zip(instance.weights, variables, strict=True)) <= instance.capacity
    )
    model.add(sum(v * x for v, x in zip(instance.values, variables, strict=True)) == target)
    differences = [
        1 - variable if bit else variable for bit, variable in zip(selected, variables, strict=True)
    ]
    model.minimize(sum(differences))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(solve_seed)
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise ValueError("Could not prove a closest optimal knapsack solution")
    return float(solver.objective_value / instance.item_count)


def _state_key(selected: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in selected)


def build_success_feature_rows(
    *,
    instances: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    references: list[dict[str, Any]],
    assignments: dict[str, str],
    history_window_budget: int,
    epsilon: float,
    distance_seed: int,
) -> list[dict[str, Any]]:
    """Build one causal feature row per non-terminal checkpoint."""
    if history_window_budget <= 0:
        raise ValueError("history_window_budget must be positive")
    instance_by_id = {
        row["instance_id"]: KnapsackInstance.from_dict(json.loads(row["instance_json"]))
        for row in instances
    }
    reference_by_id = {row["instance_id"]: row for row in references}
    trial_by_id = {row["trial_id"]: row for row in trials}
    points_by_trial: dict[str, list[dict[str, Any]]] = {}
    for point in checkpoints:
        points_by_trial.setdefault(point["trial_id"], []).append(point)
    rows: list[dict[str, Any]] = []
    distance_cache: dict[tuple[str, tuple[int, ...]], float] = {}
    for trial_id, points in sorted(points_by_trial.items()):
        trial = trial_by_id[trial_id]
        instance_id = trial["instance_id"]
        reference = reference_by_id.get(instance_id)
        if not reference or not reference["optimality_proven"]:
            continue
        optimal = float(reference["optimal_value"])
        instance = instance_by_id[instance_id]
        ordered = sorted(points, key=lambda point: point["checkpoint_index"])
        history: list[tuple[int, float, tuple[int, ...]]] = []
        best = -math.inf
        last_best_budget = 0
        best_updates = 0
        repeated_visits = 0
        visit_counts: Counter[str] = Counter()
        for point in ordered:
            payload = json.loads(point["state_json"])
            selected = tuple(int(bit) for bit in payload["selected"])
            budget_used = int(point["budget_used"])
            objective = float(point["objective_value"])
            key = _state_key(selected)
            if visit_counts[key]:
                repeated_visits += 1
            visit_counts[key] += 1
            if objective > best + epsilon:
                best = objective
                last_best_budget = budget_used
                best_updates += int(bool(history))
            history.append((budget_used, objective, selected))
            if point["is_terminal"]:
                continue
            start = history[0]
            for candidate in history:
                if candidate[0] >= budget_used - history_window_budget:
                    start = candidate
                    break
            budget_delta = budget_used - start[0]
            progress = (objective - start[1]) / max(abs(optimal), epsilon)
            progress_rate = progress / max(budget_delta / int(trial["budget_limit"]), epsilon)
            visit_probabilities = [count / len(history) for count in visit_counts.values()]
            visit_entropy = _entropy(visit_probabilities)
            if len(history) > 1:
                visit_entropy /= math.log(len(history))
            distance_key = (instance_id, selected)
            if distance_key not in distance_cache:
                distance_cache[distance_key] = nearest_optimal_distance(
                    instance,
                    selected,
                    optimal,
                    solve_seed=distance_seed,
                )
            local = neighborhood_features(
                instance,
                selected,
                objective_scale=optimal,
                epsilon=epsilon,
            )
            rows.append(
                {
                    "analysis_version": "1",
                    "checkpoint_id": point["checkpoint_id"],
                    "trial_id": trial_id,
                    "instance_id": instance_id,
                    "split": assignments[instance_id],
                    "search_method_name": trial["search_method_name"],
                    "success": bool(trial["success"]),
                    "trial_weight": 0.0,
                    "remaining_budget_fraction": float(point["remaining_budget"])
                    / int(trial["budget_limit"]),
                    "relative_objective_gap": float(point["optimality_gap"]),
                    "M2_distance": distance_cache[distance_key],
                    **local,
                    "M3_progress_rate": progress_rate,
                    "M3_stagnation": (budget_used - last_best_budget) / int(trial["budget_limit"]),
                    "M3_revisit": repeated_visits / max(len(history) - 1, 1),
                    "M3_best_update_rate": best_updates / max(budget_used, 1),
                    "M3_visit_entropy": visit_entropy,
                }
            )
    counts = Counter(row["trial_id"] for row in rows)
    for row in rows:
        row["trial_weight"] = 1.0 / counts[row["trial_id"]]
    return rows


def _random_feasible_state(instance: KnapsackInstance, rng: random.Random) -> tuple[int, ...]:
    order = list(range(instance.item_count))
    rng.shuffle(order)
    selected = [0] * instance.item_count
    weight = 0
    for index in order:
        if rng.random() < 0.5 and weight + instance.weights[index] <= instance.capacity:
            selected[index] = 1
            weight += instance.weights[index]
    return tuple(selected)


def build_instance_feature_rows(
    *,
    instances: list[dict[str, Any]],
    references: list[dict[str, Any]],
    sample_seed: int,
    sample_count: int,
    random_walk_steps: int,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Measure problem-level descriptors separately from checkpoint predictors."""
    if sample_count <= 1 or random_walk_steps <= 1:
        raise ValueError("instance sampling counts must exceed one")
    reference_by_id = {row["instance_id"]: row for row in references}
    output: list[dict[str, Any]] = []
    for instance_index, source in enumerate(instances):
        instance = KnapsackInstance.from_dict(json.loads(source["instance_json"]))
        reference = reference_by_id[source["instance_id"]]
        optimal = float(reference["optimal_value"])
        rng = random.Random(sample_seed + instance_index)
        states = [_random_feasible_state(instance, rng) for _ in range(sample_count)]
        objectives = [
            float(sum(v for v, bit in zip(instance.values, state, strict=True) if bit))
            for state in states
        ]
        distances = [
            nearest_optimal_distance(instance, state, optimal, solve_seed=sample_seed)
            for state in states
        ]
        local = [
            neighborhood_features(instance, state, objective_scale=optimal, epsilon=epsilon)
            for state in states
        ]
        walk = [states[0]]
        walk_values = [objectives[0]]
        for _ in range(random_walk_steps - 1):
            current = walk[-1]
            actions = list(range(instance.item_count))
            rng.shuffle(actions)
            next_state = current
            weight = sum(w for w, bit in zip(instance.weights, current, strict=True) if bit)
            for action in actions:
                candidate_weight = weight + (
                    -instance.weights[action] if current[action] else instance.weights[action]
                )
                if candidate_weight <= instance.capacity:
                    mutable = list(current)
                    mutable[action] = 1 - mutable[action]
                    next_state = tuple(mutable)
                    break
            walk.append(next_state)
            walk_values.append(
                float(sum(v for v, bit in zip(instance.values, next_state, strict=True) if bit))
            )
        ratios = [v / w for w, v in zip(instance.weights, instance.values, strict=True)]
        dominated = sum(
            any(
                j != i
                and instance.weights[j] <= instance.weights[i]
                and instance.values[j] >= instance.values[i]
                and (
                    instance.weights[j] < instance.weights[i]
                    or instance.values[j] > instance.values[i]
                )
                for j in range(instance.item_count)
            )
            for i in range(instance.item_count)
        )
        improving = [row["M2_improving_fraction"] for row in local]
        entropies = [row["M2_local_entropy"] for row in local]
        output.append(
            {
                "analysis_version": "1",
                "instance_id": source["instance_id"],
                "problem_size": instance.item_count,
                "capacity_ratio": instance.capacity / sum(instance.weights),
                "weight_mean": float(np.mean(instance.weights)),
                "weight_cv": _coefficient_of_variation(instance.weights),
                "value_mean": float(np.mean(instance.values)),
                "value_cv": _coefficient_of_variation(instance.values),
                "weight_value_correlation": _pearson(
                    [float(value) for value in instance.weights],
                    [float(value) for value in instance.values],
                ),
                "value_weight_ratio_mean": float(np.mean(ratios)),
                "value_weight_ratio_cv": _coefficient_of_variation(ratios),
                "value_weight_ratio_min": min(ratios),
                "value_weight_ratio_max": max(ratios),
                "dominated_item_fraction": dominated / instance.item_count,
                "sample_objective_mean": float(np.mean(objectives)),
                "sample_objective_variance": float(np.var(objectives)),
                "fitness_distance_correlation": _pearson(objectives, distances),
                "random_walk_autocorrelation": _pearson(walk_values[:-1], walk_values[1:]),
                "sample_local_optimum_rate": sum(value <= epsilon for value in improving)
                / sample_count,
                "improving_fraction_mean": float(np.mean(improving)),
                "improving_fraction_variance": float(np.var(improving)),
                "local_entropy_mean": float(np.mean(entropies)),
                "local_entropy_variance": float(np.var(entropies)),
                "sampling_seed": sample_seed + instance_index,
                "sample_count": sample_count,
                "random_walk_steps": random_walk_steps,
                "neighborhood": "one_bit_flip",
                "calculation_version": "1",
            }
        )
    return output
