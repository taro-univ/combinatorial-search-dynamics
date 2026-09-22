"""The Phase 2 Hamming-distance state representation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Self

STATE_MODEL_VERSION = "1"


class HammingDistanceStateModel:
    """Map the current objective to discrete states 0..bit_count."""

    def __init__(self, bit_count: int) -> None:
        if bit_count <= 0:
            raise ValueError("bit_count must be positive")
        self.bit_count = int(bit_count)
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
            raise ValueError("State model fitting is restricted to the train split")
        allowed = set(instance_ids)
        observed = {str(row["instance_id"]) for row in train_data}
        if not observed <= allowed:
            raise ValueError("Training data contains instances outside the declared train split")
        self._validate_distances(train_data)
        self.fit_metadata = {
            "fit_split": fit_split,
            "split_hash": split_hash,
            "instance_ids": sorted(allowed),
        }
        return self

    def transform(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fit_metadata is None:
            raise RuntimeError("State model must be fit before transform")
        self._validate_distances(data)
        return [{**row, "discrete_state": int(row["hamming_distance"])} for row in data]

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
            raise RuntimeError("Cannot save an unfitted state model")
        payload = {
            "version": STATE_MODEL_VERSION,
            "model": "hamming_distance",
            "config": {"bit_count": self.bit_count},
            "fit_metadata": self.fit_metadata,
        }
        return _atomic_json(Path(path), payload)

    @classmethod
    def load(cls, path: Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != STATE_MODEL_VERSION:
            raise ValueError(f"Unsupported state model version: {payload.get('version')!r}")
        if payload.get("model") != "hamming_distance":
            raise ValueError(f"Unsupported state model: {payload.get('model')!r}")
        model = cls(bit_count=int(payload["config"]["bit_count"]))
        model.fit_metadata = payload["fit_metadata"]
        return model

    def _validate_distances(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            distance = row.get("hamming_distance")
            if not isinstance(distance, int) or isinstance(distance, bool):
                raise TypeError("hamming_distance must be an integer")
            if distance < 0 or distance > self.bit_count:
                raise ValueError(f"hamming_distance must be in [0, {self.bit_count}]")


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
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
