"""Minimal state-model boundary used by the Phase 2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Self


class StateModelProtocol(Protocol):
    def fit(
        self,
        train_data: list[dict[str, Any]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> Self: ...

    def transform(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def fit_transform(
        self,
        train_data: list[dict[str, Any]],
        *,
        fit_split: str,
        split_hash: str,
        instance_ids: list[str],
    ) -> list[dict[str, Any]]: ...

    def save(self, path: Path) -> Path: ...

    @classmethod
    def load(cls, path: Path) -> Self: ...
