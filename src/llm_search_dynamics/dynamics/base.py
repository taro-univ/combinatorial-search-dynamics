"""Minimal transition-model boundary used by the Phase 2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

import numpy as np


class TransitionModelProtocol(Protocol):
    def fit(
        self,
        train_trajectories: list[list[int]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> Self: ...

    def predict_distribution(self, state: int, horizon: int) -> np.ndarray: ...

    def score(self, test_trajectories: list[list[int]]) -> dict[str, float | int]: ...

    def save(self, path: Path) -> Path: ...

    @classmethod
    def load(cls, path: Path) -> Self: ...
