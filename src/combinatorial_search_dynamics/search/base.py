"""Shared interfaces and candidate-evaluation budget accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from combinatorial_search_dynamics.tasks.base import TaskProtocol


class BudgetExhausted(RuntimeError):
    """Raised before an evaluation that would exceed the configured budget."""


@dataclass(frozen=True)
class CandidateEvaluation:
    action: int
    feasible: bool
    state: object | None
    objective: float | None


class BudgetedEvaluator:
    """Own the only budget-consuming path for inspecting candidate states."""

    def __init__(self, *, task: TaskProtocol, budget_limit: int) -> None:
        if isinstance(budget_limit, bool) or not isinstance(budget_limit, int) or budget_limit <= 0:
            raise ValueError("budget_limit must be a positive integer")
        self.task = task
        self.budget_limit = budget_limit
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self.budget_limit - self._used

    def evaluate(self, state: object, action: int) -> CandidateEvaluation:
        if self.remaining <= 0:
            raise BudgetExhausted("candidate-evaluation budget is exhausted")
        self._used += 1
        try:
            candidate = self.task.apply_action(state, action)
        except ValueError:
            return CandidateEvaluation(action, False, None, None)
        if not self.task.is_feasible(candidate):
            return CandidateEvaluation(action, False, None, None)
        return CandidateEvaluation(
            action=action,
            feasible=True,
            state=candidate,
            objective=float(self.task.objective(candidate)),
        )


@dataclass(frozen=True)
class SearchCheckpoint:
    checkpoint_index: int
    budget_used: int
    decision_step: int
    accepted_moves: int
    rejected_moves: int
    state: object
    objective: float
    remaining_budget: int
    is_terminal: bool
    action: int | None
    action_accepted: bool | None


@dataclass(frozen=True)
class SearchTrial:
    checkpoints: tuple[SearchCheckpoint, ...]
    success: bool
    terminal_class: str
    status: str
    error_type: str | None
    runtime_seconds: float


class SearchAlgorithm(Protocol):
    name: str
    revision: str

    def parameters(self) -> dict[str, object]: ...

    def run(
        self,
        *,
        task: TaskProtocol,
        initial_state: object,
        evaluator: BudgetedEvaluator,
        search_seed: int,
        reference_value: float | None,
    ) -> SearchTrial: ...
