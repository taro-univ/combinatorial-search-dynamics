"""Minimal classical-search implementations for the first trajectory experiment."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from combinatorial_search_dynamics.search.base import (
    BudgetedEvaluator,
    CandidateEvaluation,
    SearchCheckpoint,
    SearchTrial,
)
from combinatorial_search_dynamics.tasks.base import TaskProtocol


def _actions(task: TaskProtocol, state: object) -> list[int]:
    """Return every one-bit proposal, including currently infeasible additions."""
    if hasattr(state, "selected"):
        return list(range(len(state.selected)))
    if hasattr(state, "bits"):
        return list(range(len(state.bits)))
    return list(task.legal_actions(state))


def _improvement(task: TaskProtocol, candidate: float, current: float) -> float:
    return candidate - current if task.objective_sense == "maximize" else current - candidate


class _Recorder:
    def __init__(
        self,
        *,
        task: TaskProtocol,
        evaluator: BudgetedEvaluator,
        state: object,
        reference_value: float | None,
    ) -> None:
        self.task = task
        self.evaluator = evaluator
        self.state = state
        self.reference_value = reference_value
        self.decision_step = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.checkpoints: list[SearchCheckpoint] = []

    def success(self) -> bool:
        return self.task.is_success(self.state, self.reference_value)

    def record(
        self,
        *,
        terminal: bool,
        action: int | None,
        action_accepted: bool | None,
    ) -> None:
        self.checkpoints.append(
            SearchCheckpoint(
                checkpoint_index=len(self.checkpoints),
                budget_used=self.evaluator.used,
                decision_step=self.decision_step,
                accepted_moves=self.accepted_moves,
                rejected_moves=self.rejected_moves,
                state=self.state,
                objective=float(self.task.objective(self.state)),
                remaining_budget=self.evaluator.remaining,
                is_terminal=terminal,
                action=action,
                action_accepted=action_accepted,
            )
        )

    def accept(self, candidate: CandidateEvaluation) -> None:
        if not candidate.feasible or candidate.state is None:
            raise ValueError("cannot accept an infeasible candidate")
        self.state = candidate.state
        self.accepted_moves += 1

    def finish(self, terminal_class: str, started: float) -> SearchTrial:
        if not self.checkpoints or not self.checkpoints[-1].is_terminal:
            self.record(terminal=True, action=None, action_accepted=None)
        success = self.success()
        return SearchTrial(
            checkpoints=tuple(self.checkpoints),
            success=success,
            terminal_class="success" if success else terminal_class,
            status="completed",
            error_type=None,
            runtime_seconds=time.perf_counter() - started,
        )


@dataclass(frozen=True)
class RandomizedFirstImprovement:
    name: str = "randomized_first_improvement"
    revision: str = "1"

    def parameters(self) -> dict[str, object]:
        return {"accept_equal": False, "restart": False}

    def run(
        self,
        *,
        task: TaskProtocol,
        initial_state: object,
        evaluator: BudgetedEvaluator,
        search_seed: int,
        reference_value: float | None,
    ) -> SearchTrial:
        started = time.perf_counter()
        rng = random.Random(search_seed)
        trace = _Recorder(
            task=task,
            evaluator=evaluator,
            state=initial_state,
            reference_value=reference_value,
        )
        trace.record(terminal=trace.success(), action=None, action_accepted=None)
        if trace.success():
            return trace.finish("success", started)
        while evaluator.remaining:
            actions = _actions(task, trace.state)
            rng.shuffle(actions)
            accepted: CandidateEvaluation | None = None
            last_action: int | None = None
            for action in actions:
                if not evaluator.remaining:
                    break
                candidate = evaluator.evaluate(trace.state, action)
                last_action = action
                if (
                    candidate.feasible
                    and candidate.objective is not None
                    and task.is_better(candidate.objective, task.objective(trace.state))
                ):
                    accepted = candidate
                    break
                trace.rejected_moves += 1
            trace.decision_step += 1
            if accepted is None:
                terminal_class = "budget_exhausted" if not evaluator.remaining else "local_optimum"
                trace.record(terminal=True, action=last_action, action_accepted=False)
                return trace.finish(terminal_class, started)
            trace.accept(accepted)
            terminal = trace.success() or not evaluator.remaining
            trace.record(terminal=terminal, action=accepted.action, action_accepted=True)
            if terminal:
                return trace.finish("budget_exhausted", started)
        return trace.finish("budget_exhausted", started)


@dataclass(frozen=True)
class SimulatedAnnealing:
    initial_temperature: float
    final_temperature: float
    name: str = "simulated_annealing"
    revision: str = "1"

    def __post_init__(self) -> None:
        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("annealing temperatures must be positive")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must not exceed initial_temperature")

    def parameters(self) -> dict[str, object]:
        return {
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "schedule": "geometric_by_budget",
            "restart": False,
        }

    def run(
        self,
        *,
        task: TaskProtocol,
        initial_state: object,
        evaluator: BudgetedEvaluator,
        search_seed: int,
        reference_value: float | None,
    ) -> SearchTrial:
        started = time.perf_counter()
        rng = random.Random(search_seed)
        trace = _Recorder(
            task=task,
            evaluator=evaluator,
            state=initial_state,
            reference_value=reference_value,
        )
        trace.record(terminal=trace.success(), action=None, action_accepted=None)
        if trace.success():
            return trace.finish("success", started)
        while evaluator.remaining:
            trace.decision_step += 1
            action = rng.choice(_actions(task, trace.state))
            current = float(task.objective(trace.state))
            candidate = evaluator.evaluate(trace.state, action)
            accepted = False
            if candidate.feasible and candidate.objective is not None:
                gain = _improvement(task, candidate.objective, current)
                progress = evaluator.used / evaluator.budget_limit
                temperature = (
                    self.initial_temperature
                    * (self.final_temperature / self.initial_temperature) ** progress
                )
                accepted = gain >= 0 or rng.random() < math.exp(gain / temperature)
            if accepted:
                trace.accept(candidate)
            else:
                trace.rejected_moves += 1
            terminal = trace.success() or not evaluator.remaining
            trace.record(terminal=terminal, action=action, action_accepted=accepted)
            if terminal:
                return trace.finish("budget_exhausted", started)
        return trace.finish("budget_exhausted", started)


@dataclass(frozen=True)
class ShortTermTabuSearch:
    tenure: int
    name: str = "short_term_tabu"
    revision: str = "1"

    def __post_init__(self) -> None:
        if isinstance(self.tenure, bool) or not isinstance(self.tenure, int) or self.tenure <= 0:
            raise ValueError("tabu tenure must be a positive integer")

    def parameters(self) -> dict[str, object]:
        return {
            "tenure": self.tenure,
            "aspiration": "global_best",
            "long_term_memory": False,
            "restart": False,
        }

    def run(
        self,
        *,
        task: TaskProtocol,
        initial_state: object,
        evaluator: BudgetedEvaluator,
        search_seed: int,
        reference_value: float | None,
    ) -> SearchTrial:
        started = time.perf_counter()
        rng = random.Random(search_seed)
        trace = _Recorder(
            task=task,
            evaluator=evaluator,
            state=initial_state,
            reference_value=reference_value,
        )
        best_objective = float(task.objective(initial_state))
        tabu_until: dict[int, int] = {}
        trace.record(terminal=trace.success(), action=None, action_accepted=None)
        if trace.success():
            return trace.finish("success", started)
        while evaluator.remaining:
            trace.decision_step += 1
            candidates: list[CandidateEvaluation] = []
            actions = _actions(task, trace.state)
            rng.shuffle(actions)
            for action in actions:
                if not evaluator.remaining:
                    break
                candidate = evaluator.evaluate(trace.state, action)
                if not candidate.feasible or candidate.objective is None:
                    trace.rejected_moves += 1
                    continue
                is_tabu = tabu_until.get(action, 0) >= trace.decision_step
                aspiration = task.is_better(candidate.objective, best_objective)
                if is_tabu and not aspiration:
                    trace.rejected_moves += 1
                    continue
                candidates.append(candidate)
            if not candidates:
                terminal_class = (
                    "budget_exhausted" if not evaluator.remaining else "no_admissible_move"
                )
                trace.record(terminal=True, action=None, action_accepted=False)
                return trace.finish(terminal_class, started)
            chosen = candidates[0]
            for candidate in candidates[1:]:
                if task.is_better(float(candidate.objective), float(chosen.objective)):
                    chosen = candidate
            trace.rejected_moves += len(candidates) - 1
            trace.accept(chosen)
            tabu_until[chosen.action] = trace.decision_step + self.tenure
            if task.is_better(float(chosen.objective), best_objective):
                best_objective = float(chosen.objective)
            terminal = trace.success() or not evaluator.remaining
            trace.record(terminal=terminal, action=chosen.action, action_accepted=True)
            if terminal:
                return trace.finish("budget_exhausted", started)
        return trace.finish("budget_exhausted", started)
