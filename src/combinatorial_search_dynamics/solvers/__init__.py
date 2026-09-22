"""Reference-solver adapters, separate from search generators."""

from combinatorial_search_dynamics.solvers.base import SolverParameters, SolverResult, SolverStatus
from combinatorial_search_dynamics.solvers.registry import get_solver

__all__ = ["SolverParameters", "SolverResult", "SolverStatus", "get_solver"]
