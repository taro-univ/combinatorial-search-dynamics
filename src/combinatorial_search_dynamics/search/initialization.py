"""Deterministic initial-state construction shared across search methods."""

from __future__ import annotations

import random

from combinatorial_search_dynamics.tasks.knapsack import (
    KnapsackInstance,
    KnapsackState,
    KnapsackTask,
)


def make_initial_state(
    *,
    task: object,
    instance: object,
    seed: int,
    strategy: str,
    include_probability: float,
) -> object:
    if strategy == "task_default":
        return task.initial_state(instance)
    if strategy != "random_feasible":
        raise ValueError(f"Unknown initial-state strategy: {strategy}")
    if not isinstance(task, KnapsackTask) or not isinstance(instance, KnapsackInstance):
        return task.initial_state(instance)
    if not 0.0 <= include_probability <= 1.0:
        raise ValueError("include_probability must be between 0 and 1")
    rng = random.Random(seed)
    order = list(range(instance.item_count))
    rng.shuffle(order)
    selected = [0] * instance.item_count
    total_weight = 0
    total_value = 0
    for index in order:
        if rng.random() >= include_probability:
            continue
        if total_weight + instance.weights[index] <= instance.capacity:
            selected[index] = 1
            total_weight += instance.weights[index]
            total_value += instance.values[index]
    return KnapsackState(
        selected=tuple(selected),
        weights=instance.weights,
        values=instance.values,
        capacity=instance.capacity,
        total_weight=total_weight,
        total_value=total_value,
        step=0,
        max_steps=task.max_steps,
    )
