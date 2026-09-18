"""Seeded mock search generator; this module does not call an LLM."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from llm_search_dynamics.identifiers import checkpoint_id, trial_id
from llm_search_dynamics.tasks.base import TaskProtocol

GENERATOR_NAME = "mock_binary_search"
GENERATOR_REVISION = "1"


def generator_name(task_name: str) -> str:
    """Keep the Phase 2 name stable while identifying the knapsack mock clearly."""
    return GENERATOR_NAME if task_name == "dummy_binary" else f"mock_{task_name}_search"


@dataclass(frozen=True)
class GeneratedCheckpoint:
    checkpoint_id: str
    trial_id: str
    generated_token_index: int
    checkpoint_index: int
    state: object
    objective: float
    remaining_budget: int
    is_terminal: bool
    action: int | None


@dataclass(frozen=True)
class GeneratedTrial:
    trial_id: str
    checkpoints: tuple[GeneratedCheckpoint, ...]
    success: bool
    terminal_class: str
    status: str
    error_type: str | None
    runtime_seconds: float


class MockGenerator:
    def __init__(self, *, greedy_probability: float, max_steps: int) -> None:
        if not 0.0 <= greedy_probability <= 1.0:
            raise ValueError("greedy_probability must be between 0 and 1")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.greedy_probability = greedy_probability
        self.max_steps = max_steps

    def generate(
        self,
        *,
        task: TaskProtocol,
        instance: object,
        source_instance_id: str,
        experiment_conditions: dict[str, object],
        sampling_seed: int,
        reference_value: float | None = None,
    ) -> GeneratedTrial:
        rng = random.Random(sampling_seed)
        identifier = trial_id(source_instance_id, experiment_conditions, sampling_seed)
        started = time.perf_counter()
        state = task.initial_state(instance)
        checkpoints: list[GeneratedCheckpoint] = []

        def record(action: int | None) -> None:
            position = len(checkpoints)
            checkpoints.append(
                GeneratedCheckpoint(
                    checkpoint_id=checkpoint_id(identifier, position),
                    trial_id=identifier,
                    generated_token_index=position,
                    checkpoint_index=position,
                    state=state,
                    objective=task.objective(state),
                    remaining_budget=max(0, self.max_steps - position),
                    is_terminal=task.is_terminal(state, reference_value)
                    or position >= self.max_steps,
                    action=action,
                )
            )

        record(None)
        try:
            while (
                not task.is_terminal(state, reference_value)
                and len(checkpoints) - 1 < self.max_steps
            ):
                actions = task.legal_actions(state)
                if not actions:
                    return GeneratedTrial(
                        identifier,
                        tuple(checkpoints),
                        False,
                        "failed",
                        "failed",
                        "no_legal_actions",
                        time.perf_counter() - started,
                    )
                if rng.random() < self.greedy_probability:
                    scored = [
                        (task.objective(task.apply_action(state, action)), action)
                        for action in actions
                    ]
                    best = scored[0][0]
                    candidates = [scored[0][1]]
                    for score, candidate_action in scored[1:]:
                        if task.is_better(score, best):
                            best = score
                            candidates = [candidate_action]
                        elif abs(score - best) <= 1e-9:
                            candidates.append(candidate_action)
                    action = rng.choice(candidates)
                else:
                    action = rng.choice(actions)
                state = task.apply_action(state, action)
                record(action)
        except Exception as exc:  # noqa: BLE001 - preserve failed trials, only error type is stored
            return GeneratedTrial(
                identifier,
                tuple(checkpoints),
                False,
                "failed",
                "failed",
                type(exc).__name__,
                time.perf_counter() - started,
            )

        success = task.is_success(state, reference_value)
        return GeneratedTrial(
            trial_id=identifier,
            checkpoints=tuple(checkpoints),
            success=success,
            terminal_class="success" if success else "budget_exhausted",
            status="completed",
            error_type=None,
            runtime_seconds=time.perf_counter() - started,
        )
