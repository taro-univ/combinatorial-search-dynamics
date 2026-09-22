"""Helpers for absorbing discrete states."""

from __future__ import annotations

import numpy as np


def enforce_absorbing_state(matrix: np.ndarray, state: int = 0) -> None:
    """Mutate one row to be a deterministic self-transition."""
    if state < 0 or state >= matrix.shape[0]:
        raise ValueError("Absorbing state is outside the transition matrix")
    matrix[state, :] = 0.0
    matrix[state, state] = 1.0
