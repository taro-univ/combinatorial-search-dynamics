from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Schedule(Protocol):
    def __call__(self, step: int, total_steps: int) -> tuple[float, float]: ...


@dataclass(frozen=True)
class LinearSchedule:
    a_start: float
    a_end: float
    t_start: float
    t_end: float
    name: str = "linear_path"

    def __call__(self, step: int, total_steps: int) -> tuple[float, float]:
        if step <= 0:
            return float(self.a_start), float(max(0.0, self.t_start))
        if total_steps <= 1 or step >= total_steps - 1:
            return float(self.a_end), float(max(0.0, self.t_end))
        fraction = 1.0 if total_steps <= 1 else np.clip(step / (total_steps - 1), 0.0, 1.0)
        a = self.a_start + fraction * (self.a_end - self.a_start)
        temperature = self.t_start + fraction * (self.t_end - self.t_start)
        return float(a), float(max(0.0, temperature))


@dataclass(frozen=True)
class PiecewiseLinearSchedule:
    """Open-loop (a,T) path; deliberately has no state-feedback interface."""

    points: Sequence[tuple[float, float, float]]
    name: str = "piecewise_path"

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("At least two (fraction, a, T) points are required")
        fractions = [point[0] for point in self.points]
        if fractions[0] != 0.0 or fractions[-1] != 1.0 or fractions != sorted(fractions):
            raise ValueError("Path fractions must be sorted and span 0 to 1")
        if any(point[2] < 0 for point in self.points):
            raise ValueError("Temperature must be nonnegative")

    def __call__(self, step: int, total_steps: int) -> tuple[float, float]:
        fraction = 1.0 if total_steps <= 1 else np.clip(step / (total_steps - 1), 0.0, 1.0)
        fractions = np.asarray([point[0] for point in self.points])
        gains = np.asarray([point[1] for point in self.points])
        temperatures = np.asarray([point[2] for point in self.points])
        return float(np.interp(fraction, fractions, gains)), float(
            np.interp(fraction, fractions, temperatures)
        )


def schedule_from_dict(values: dict[str, object]) -> LinearSchedule:
    kind = str(values.get("kind", "path"))
    return LinearSchedule(
        a_start=float(values["a_start"]),
        a_end=float(values["a_end"]),
        t_start=float(values["t_start"]),
        t_end=float(values["t_end"]),
        name=kind,
    )
