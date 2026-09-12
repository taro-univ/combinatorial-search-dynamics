"""First-order Markov transition model for discrete Hamming states."""

from __future__ import annotations

import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np

from llm_search_dynamics.dynamics.absorbing import enforce_absorbing_state

DYNAMICS_MODEL_VERSION = "1"


class MarkovTransitionModel:
    def __init__(self, state_count: int, smoothing: float = 0.0) -> None:
        if state_count <= 0:
            raise ValueError("state_count must be positive")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.state_count = int(state_count)
        self.smoothing = float(smoothing)
        self.transition_matrix: np.ndarray | None = None
        self.fit_metadata: dict[str, Any] | None = None

    def fit(
        self,
        train_trajectories: list[list[int]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> Self:
        if fit_split != "train":
            raise ValueError("Dynamics fitting is restricted to the train split")
        counts = np.zeros((self.state_count, self.state_count), dtype=np.float64)
        for trajectory in train_trajectories:
            self._validate_trajectory(trajectory)
            for current, following in pairwise(trajectory):
                counts[current, following] += 1.0

        matrix = np.zeros_like(counts)
        for state in range(self.state_count):
            if state == 0:
                continue
            row = counts[state] + self.smoothing
            total = float(row.sum())
            if total == 0.0:
                matrix[state, state] = 1.0
            else:
                matrix[state] = row / total
        enforce_absorbing_state(matrix)
        self.transition_matrix = matrix
        self.fit_metadata = {
            "fit_split": fit_split,
            "split_hash": split_hash,
            "instance_ids": sorted(set(instance_ids)),
        }
        return self

    def predict_distribution(self, state: int, horizon: int) -> np.ndarray:
        matrix = self._require_fit()
        self._validate_state(state)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        return np.linalg.matrix_power(matrix, horizon)[state].copy()

    def score(self, test_trajectories: list[list[int]]) -> dict[str, float | int]:
        matrix = self._require_fit()
        losses: list[float] = []
        correct = 0
        for trajectory in test_trajectories:
            self._validate_trajectory(trajectory)
            for current, following in pairwise(trajectory):
                probability = max(float(matrix[current, following]), 1e-12)
                losses.append(-float(np.log(probability)))
                correct += int(int(np.argmax(matrix[current])) == following)
        return {
            "negative_log_likelihood": float(np.mean(losses)) if losses else 0.0,
            "next_state_accuracy": correct / len(losses) if losses else 0.0,
            "n_transitions": len(losses),
        }

    def save(self, path: Path) -> Path:
        matrix = self._require_fit()
        payload = {
            "version": DYNAMICS_MODEL_VERSION,
            "model": "first_order_markov",
            "state_count": self.state_count,
            "smoothing": self.smoothing,
            "fit_metadata": self.fit_metadata,
            "transition_matrix": matrix.tolist(),
        }
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Model artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != DYNAMICS_MODEL_VERSION:
            raise ValueError(f"Unsupported dynamics model version: {payload.get('version')!r}")
        model = cls(
            state_count=int(payload["state_count"]),
            smoothing=float(payload["smoothing"]),
        )
        matrix = np.asarray(payload["transition_matrix"], dtype=np.float64)
        if matrix.shape != (model.state_count, model.state_count):
            raise ValueError("Saved transition matrix has an invalid shape")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("Saved transition matrix is not row stochastic")
        model.transition_matrix = matrix
        model.fit_metadata = payload["fit_metadata"]
        return model

    def _require_fit(self) -> np.ndarray:
        if self.transition_matrix is None:
            raise RuntimeError("Dynamics model must be fit before use")
        return self.transition_matrix

    def _validate_state(self, state: int) -> None:
        if (
            not isinstance(state, int)
            or isinstance(state, bool)
            or not 0 <= state < self.state_count
        ):
            raise ValueError(f"state must be an integer in [0, {self.state_count - 1}]")

    def _validate_trajectory(self, trajectory: list[int]) -> None:
        if not trajectory:
            raise ValueError("Trajectories must not be empty")
        for state in trajectory:
            self._validate_state(state)
