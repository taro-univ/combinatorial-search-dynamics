"""A priori bins of the current proven-optimum relative objective gap."""

from __future__ import annotations

import bisect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Self

GAP_MODEL_VERSION = "1"


class ObjectiveGapStateModel:
    """State 0 is proven-optimal; positive gaps map to consecutive bins."""

    def __init__(self, *, bin_boundaries: list[float]) -> None:
        boundaries = [float(value) for value in bin_boundaries]
        if not boundaries or any(not 0 < value < 1 for value in boundaries):
            raise ValueError("bin boundaries must be non-empty and strictly between 0 and 1")
        if boundaries != sorted(set(boundaries)):
            raise ValueError("bin boundaries must be strictly increasing")
        self.bin_boundaries = tuple(boundaries)
        self.state_count = len(boundaries) + 2
        self.fit_metadata: dict[str, Any] | None = None

    def fit(
        self,
        train_data: list[dict[str, Any]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> Self:
        if fit_split != "train":
            raise ValueError("Objective gap model fitting is restricted to train")
        if not train_data:
            raise ValueError("No proven-optimum train checkpoints are available")
        allowed = set(instance_ids)
        if not {row["instance_id"] for row in train_data} <= allowed:
            raise ValueError("Training data contains an instance outside the train split")
        self._validate_gaps(train_data)
        self.fit_metadata = {
            "fit_split": fit_split,
            "split_hash": split_hash,
            "instance_ids": sorted({row["instance_id"] for row in train_data}),
            "excluded_train_instance_ids": sorted(
                allowed - {row["instance_id"] for row in train_data}
            ),
        }
        return self

    def transform(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fit_metadata is None:
            raise RuntimeError("Objective gap state model must be fit before transform")
        self._validate_gaps(data)
        transformed = []
        for row in data:
            gap = float(row["optimality_gap"])
            state = 0 if gap <= 1e-9 else bisect.bisect_left(self.bin_boundaries, gap) + 1
            transformed.append({**row, "discrete_state": state})
        return transformed

    def fit_transform(
        self,
        train_data: list[dict[str, Any]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> list[dict[str, Any]]:
        return self.fit(
            train_data,
            fit_split=fit_split,
            split_hash=split_hash,
            instance_ids=instance_ids,
        ).transform(train_data)

    def save(self, path: Path) -> Path:
        if self.fit_metadata is None:
            raise RuntimeError("Cannot save an unfitted objective gap model")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"State model artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GAP_MODEL_VERSION,
            "model": "objective_gap",
            "config": {"bin_boundaries": list(self.bin_boundaries)},
            "fit_metadata": self.fit_metadata,
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != GAP_MODEL_VERSION or payload.get("model") != "objective_gap":
            raise ValueError("Unsupported objective gap state model artifact")
        model = cls(bin_boundaries=payload["config"]["bin_boundaries"])
        model.fit_metadata = payload["fit_metadata"]
        return model

    @staticmethod
    def _validate_gaps(data: list[dict[str, Any]]) -> None:
        for row in data:
            value = row.get("optimality_gap")
            if value is None:
                raise ValueError("A proven optimality gap is required")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Proven optimality gap must be numeric")
            if not 0 <= value <= 1 + 1e-9:
                raise ValueError("Relative optimality gap must be in [0, 1]")
