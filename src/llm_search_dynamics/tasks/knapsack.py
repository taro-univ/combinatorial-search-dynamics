"""Deterministic single-capacity 0-1 knapsack task with immutable states."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

OBJECTIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class KnapsackInstance:
    weights: tuple[int, ...]
    values: tuple[int, ...]
    capacity: int
    item_count: int
    generation_seed: int
    task_name: str = "knapsack"
    task_version: str = "1"

    def validate_structure(self) -> None:
        """Validate stored instance facts without generation-time settings."""
        if self.task_name != "knapsack" or self.task_version != "1":
            raise ValueError("instance task name or version is unsupported")
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or self.item_count < 2
        ):
            raise ValueError("item_count must be an integer of at least 2")
        if len(self.weights) != self.item_count or len(self.values) != self.item_count:
            raise ValueError("weight/value lengths must equal item_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (*self.weights, *self.values)
        ):
            raise ValueError("weights and values must be positive integers")
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        if isinstance(self.generation_seed, bool) or not isinstance(self.generation_seed, int):
            raise TypeError("generation_seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "values": list(self.values),
            "capacity": self.capacity,
            "item_count": self.item_count,
            "generation_seed": self.generation_seed,
            "task_name": self.task_name,
            "task_version": self.task_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KnapsackInstance:
        return cls(
            weights=tuple(payload["weights"]),
            values=tuple(payload["values"]),
            capacity=payload["capacity"],
            item_count=payload["item_count"],
            generation_seed=payload["generation_seed"],
            task_name=payload["task_name"],
            task_version=payload["task_version"],
        )


@dataclass(frozen=True)
class KnapsackState:
    selected: tuple[int, ...]
    weights: tuple[int, ...]
    values: tuple[int, ...]
    capacity: int
    total_weight: int
    total_value: int
    step: int
    max_steps: int


class KnapsackTask:
    """Maximize selected value while every intermediate state remains feasible."""

    name = "knapsack"
    version = "1"
    objective_sense = "maximize"

    @classmethod
    def for_stored_instance(cls, instance: KnapsackInstance, *, max_steps: int) -> KnapsackTask:
        """Make a state checker from stored facts; generation ranges are irrelevant."""
        instance.validate_structure()
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        task = object.__new__(cls)
        task.item_count = instance.item_count
        task.min_weight = min(instance.weights)
        task.max_weight = max(instance.weights)
        task.min_value = min(instance.values)
        task.max_value = max(instance.values)
        task.capacity_ratio = instance.capacity / sum(instance.weights)
        task.max_steps = max_steps
        return task

    def __init__(
        self,
        *,
        item_count: int,
        min_weight: int,
        max_weight: int,
        min_value: int,
        max_value: int,
        capacity_ratio: float,
        max_steps: int,
    ) -> None:
        for label, value in (
            ("item_count", item_count),
            ("min_weight", min_weight),
            ("max_weight", max_weight),
            ("min_value", min_value),
            ("max_value", max_value),
            ("max_steps", max_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if item_count < 2:
            raise ValueError("item_count must be at least 2")
        if min_weight > max_weight or min_value > max_value:
            raise ValueError("minimum weight/value must not exceed maximum")
        if not 0.0 < capacity_ratio < 1.0:
            raise ValueError("capacity_ratio must be strictly between 0 and 1")
        self.item_count = item_count
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_value = min_value
        self.max_value = max_value
        self.capacity_ratio = float(capacity_ratio)
        self.max_steps = max_steps

    def generate_instance(self, seed: int, **params: Any) -> KnapsackInstance:
        del params
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("generation seed must be an integer")
        rng = random.Random(seed)
        weights = tuple(
            rng.randint(self.min_weight, self.max_weight) for _ in range(self.item_count)
        )
        values = tuple(rng.randint(self.min_value, self.max_value) for _ in range(self.item_count))
        capacity = max(1, min(sum(weights) - 1, round(sum(weights) * self.capacity_ratio)))
        instance = KnapsackInstance(
            weights, values, capacity, self.item_count, seed, self.name, self.version
        )
        self.validate_instance(instance)
        return instance

    def validate_instance(self, instance: KnapsackInstance) -> None:
        if not isinstance(instance, KnapsackInstance):
            raise TypeError("instance must be KnapsackInstance")
        instance.validate_structure()
        if instance.item_count != self.item_count:
            raise ValueError("instance item_count does not match task configuration")

    def initial_state(self, instance: KnapsackInstance) -> KnapsackState:
        self.validate_instance(instance)
        return KnapsackState(
            selected=(0,) * instance.item_count,
            weights=instance.weights,
            values=instance.values,
            capacity=instance.capacity,
            total_weight=0,
            total_value=0,
            step=0,
            max_steps=self.max_steps,
        )

    def legal_actions(self, state: KnapsackState) -> tuple[int, ...]:
        self._validate_state(state)
        return tuple(
            index
            for index, selected in enumerate(state.selected)
            if selected or state.total_weight + state.weights[index] <= state.capacity
        )

    def apply_action(self, state: KnapsackState, action: int) -> KnapsackState:
        self._validate_state(state)
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < self.item_count
        ):
            raise ValueError(f"action must be an item index in [0, {self.item_count - 1}]")
        if action not in self.legal_actions(state):
            raise ValueError("action would exceed knapsack capacity")
        selected = list(state.selected)
        direction = -1 if selected[action] else 1
        selected[action] = 1 - selected[action]
        return KnapsackState(
            selected=tuple(selected),
            weights=state.weights,
            values=state.values,
            capacity=state.capacity,
            total_weight=state.total_weight + direction * state.weights[action],
            total_value=state.total_value + direction * state.values[action],
            step=state.step + 1,
            max_steps=state.max_steps,
        )

    def is_feasible(self, state: KnapsackState) -> bool:
        try:
            self._validate_state(state)
        except (TypeError, ValueError):
            return False
        return state.total_weight <= state.capacity

    def is_terminal(self, state: KnapsackState, reference_value: float | None = None) -> bool:
        self._validate_state(state)
        return self.is_success(state, reference_value) or state.step >= state.max_steps

    def objective(self, state: KnapsackState) -> float:
        self._validate_state(state)
        return float(state.total_value)

    def is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate > incumbent + OBJECTIVE_TOLERANCE

    def objective_gap(self, value: float, reference: float, *, epsilon: float = 1e-9) -> float:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return max(0.0, reference - value) / max(abs(reference), epsilon)

    def is_success(self, state: KnapsackState, reference_value: float | None = None) -> bool:
        if reference_value is None or not self.is_feasible(state):
            return False
        return self.objective(state) >= reference_value - OBJECTIVE_TOLERANCE

    def serialize_state(self, state: KnapsackState) -> dict[str, Any]:
        self._validate_state(state)
        return {
            "selected": list(state.selected),
            "weights": list(state.weights),
            "values": list(state.values),
            "capacity": state.capacity,
            "total_weight": state.total_weight,
            "total_value": state.total_value,
            "step": state.step,
            "max_steps": state.max_steps,
        }

    def deserialize_state(self, payload: dict[str, Any]) -> KnapsackState:
        try:
            state = KnapsackState(
                selected=tuple(payload["selected"]),
                weights=tuple(payload["weights"]),
                values=tuple(payload["values"]),
                capacity=payload["capacity"],
                total_weight=payload["total_weight"],
                total_value=payload["total_value"],
                step=payload["step"],
                max_steps=payload["max_steps"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid serialized knapsack state") from exc
        self._validate_state(state)
        return state

    def parse_action(self, text: str) -> int:
        if not isinstance(text, str) or re.fullmatch(r"[0-9]+", text.strip()) is None:
            raise ValueError("action must be a decimal item index")
        action = int(text.strip())
        if not 0 <= action < self.item_count:
            raise ValueError(f"action must be in [0, {self.item_count - 1}]")
        return action

    def _validate_state(self, state: KnapsackState) -> None:
        if not isinstance(state, KnapsackState):
            raise TypeError("state must be KnapsackState")
        if (
            len(state.selected) != self.item_count
            or len(state.weights) != self.item_count
            or len(state.values) != self.item_count
        ):
            raise ValueError("state vector lengths must equal item_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in state.selected
        ):
            raise ValueError("selected vector must contain only 0 or 1")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (*state.weights, *state.values)
        ):
            raise ValueError("state weights/values must be positive integers")
        if state.capacity <= 0 or state.step < 0 or state.max_steps <= 0:
            raise ValueError("state capacity or step budget is invalid")
        weight = sum(
            weight
            for weight, selected in zip(state.weights, state.selected, strict=True)
            if selected
        )
        value = sum(
            value for value, selected in zip(state.values, state.selected, strict=True) if selected
        )
        if state.total_weight != weight:
            raise ValueError("state total_weight does not equal the selected-weight sum")
        if state.total_value != value:
            raise ValueError("state total_value does not equal the selected-value sum")
