from __future__ import annotations

import pytest

from llm_search_dynamics.search.algorithms import (
    RandomizedFirstImprovement,
    ShortTermTabuSearch,
    SimulatedAnnealing,
)
from llm_search_dynamics.search.base import BudgetedEvaluator
from llm_search_dynamics.tasks.dummy_binary import BinaryInstance, DummyBinaryTask


@pytest.mark.parametrize(
    "algorithm",
    [
        RandomizedFirstImprovement(),
        SimulatedAnnealing(initial_temperature=2.0, final_temperature=0.1),
        ShortTermTabuSearch(tenure=2),
    ],
)
def test_search_is_deterministic_and_never_exceeds_candidate_budget(algorithm: object) -> None:
    task = DummyBinaryTask(bit_count=5, max_steps=12)
    instance = BinaryInstance(5, (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))

    def run() -> tuple[object, ...]:
        evaluator = BudgetedEvaluator(task=task, budget_limit=12)
        result = algorithm.run(
            task=task,
            initial_state=task.initial_state(instance),
            evaluator=evaluator,
            search_seed=47,
            reference_value=0.0,
        )
        assert evaluator.used <= evaluator.budget_limit
        assert result.checkpoints[-1].is_terminal
        assert all(point.budget_used <= 12 for point in result.checkpoints)
        assert all(point.remaining_budget == 12 - point.budget_used for point in result.checkpoints)
        assert result.checkpoints[-1].accepted_moves + result.checkpoints[-1].rejected_moves == (
            evaluator.used
        )
        return tuple(
            (
                point.budget_used,
                point.decision_step,
                point.accepted_moves,
                point.rejected_moves,
                point.state,
                point.action,
                point.action_accepted,
            )
            for point in result.checkpoints
        )

    assert run() == run()


def test_infeasible_candidate_consumes_one_evaluation() -> None:
    task = DummyBinaryTask(bit_count=2, max_steps=2)
    state = task.initial_state(BinaryInstance(2, (0, 0), (1, 1)))
    evaluator = BudgetedEvaluator(task=task, budget_limit=1)
    candidate = evaluator.evaluate(state, 99)
    assert not candidate.feasible
    assert evaluator.used == 1
    assert evaluator.remaining == 0
