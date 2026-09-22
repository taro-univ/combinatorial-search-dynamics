"""Solver-neutral immutable result and status contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Protocol


class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    MODEL_INVALID = "MODEL_INVALID"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SolverParameters:
    max_time_seconds: float
    num_search_workers: int
    random_seed: int
    relative_gap_epsilon: float
    log_search_progress: bool

    def __post_init__(self) -> None:
        if not isfinite(self.max_time_seconds) or self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive and finite")
        if (
            isinstance(self.num_search_workers, bool)
            or not isinstance(self.num_search_workers, int)
            or self.num_search_workers <= 0
        ):
            raise ValueError("num_search_workers must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if not isfinite(self.relative_gap_epsilon) or self.relative_gap_epsilon <= 0:
            raise ValueError("relative_gap_epsilon must be positive and finite")
        if not isinstance(self.log_search_progress, bool):
            raise TypeError("log_search_progress must be boolean")

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "max_time_seconds": self.max_time_seconds,
            "num_search_workers": self.num_search_workers,
            "random_seed": self.random_seed,
            "relative_gap_epsilon": self.relative_gap_epsilon,
            "log_search_progress": self.log_search_progress,
        }


@dataclass(frozen=True)
class SolverResult:
    instance_id: str
    solver_name: str
    solver_version: str
    status: SolverStatus
    best_feasible_value: float | None
    best_bound: float | None
    optimal_value: float | None
    optimality_gap: float | None
    optimality_proven: bool
    timed_out: bool
    runtime_seconds: float
    solution: tuple[int, ...] | None
    parameters: SolverParameters
    error_type: str | None = None


class SolverProtocol(Protocol):
    name: str
    version: str
    parameters: SolverParameters

    def solve(self, instance: Any, *, instance_id: str) -> SolverResult: ...
