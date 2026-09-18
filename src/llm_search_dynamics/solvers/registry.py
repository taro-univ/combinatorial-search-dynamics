"""Small solver registry; no solver implementation leaks into TaskProtocol."""

from __future__ import annotations

from llm_search_dynamics.solvers.base import SolverParameters, SolverProtocol
from llm_search_dynamics.solvers.ortools_knapsack import OrtoolsKnapsackSolver
from llm_search_dynamics.tasks.knapsack import KnapsackTask


def get_solver(name: str, *, task: KnapsackTask, parameters: SolverParameters) -> SolverProtocol:
    if name == OrtoolsKnapsackSolver.name:
        return OrtoolsKnapsackSolver(task=task, parameters=parameters)
    raise KeyError(f"Unknown solver: {name}")
