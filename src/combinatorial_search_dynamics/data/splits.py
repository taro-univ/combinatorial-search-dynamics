"""Deterministic, instance-level dataset splitting."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from combinatorial_search_dynamics.identifiers import canonical_json

SPLIT_VERSION = "1"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitDefinition:
    """An immutable instance-to-split assignment and its content hash."""

    version: str
    seed: int
    ratios: dict[str, float]
    assignments: dict[str, str]
    split_hash: str

    def instance_ids(self, split: str) -> list[str]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        return sorted(key for key, value in self.assignments.items() if value == split)

    def fit_instance_ids(self, split: str = "train") -> list[str]:
        """Return IDs allowed for fitting; held-out test is explicitly rejected."""
        if split != "train":
            raise ValueError("Models may only be fit with the train split")
        return self.instance_ids(split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "ratios": self.ratios,
            "assignments": self.assignments,
            "split_hash": self.split_hash,
        }


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_instance_split(
    instance_ids: list[str], *, seed: int, ratios: dict[str, float]
) -> SplitDefinition:
    """Split unique instance IDs reproducibly while keeping every split non-empty."""
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Ratios must contain exactly: {', '.join(SPLIT_NAMES)}")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive numbers")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    unique_ids = sorted(set(instance_ids))
    if len(unique_ids) != len(instance_ids):
        raise ValueError("instance_ids must be unique")
    if len(unique_ids) < len(SPLIT_NAMES):
        raise ValueError("At least three instances are required for non-empty splits")

    shuffled = unique_ids.copy()
    random.Random(seed).shuffle(shuffled)

    counts = {name: 1 for name in SPLIT_NAMES}
    remaining = len(shuffled) - len(SPLIT_NAMES)
    targets = {name: ratios[name] * remaining for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        counts[name] += int(targets[name])
    unassigned = len(shuffled) - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (targets[name] - int(targets[name]), ratios[name]),
        reverse=True,
    )
    for index in range(unassigned):
        counts[order[index % len(order)]] += 1

    assignments: dict[str, str] = {}
    offset = 0
    for name in SPLIT_NAMES:
        for instance_id in shuffled[offset : offset + counts[name]]:
            assignments[instance_id] = name
        offset += counts[name]

    normalized_ratios = {name: float(ratios[name]) for name in SPLIT_NAMES}
    payload = {
        "version": SPLIT_VERSION,
        "seed": int(seed),
        "ratios": normalized_ratios,
        "assignments": dict(sorted(assignments.items())),
    }
    return SplitDefinition(
        version=SPLIT_VERSION,
        seed=int(seed),
        ratios=normalized_ratios,
        assignments=payload["assignments"],
        split_hash=_content_hash(payload),
    )


def write_split(path: Path, split: SplitDefinition, *, overwrite: bool = False) -> Path:
    """Atomically write a split definition."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Split already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(split.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_split(path: Path) -> SplitDefinition:
    """Read and verify a split definition and its content hash."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(payload.get("version", ""))
    if version != SPLIT_VERSION:
        raise ValueError(f"Unsupported split version: {version!r}")
    content = {
        "version": version,
        "seed": int(payload["seed"]),
        "ratios": payload["ratios"],
        "assignments": payload["assignments"],
    }
    expected = _content_hash(content)
    if payload.get("split_hash") != expected:
        raise ValueError("Split content hash does not match its contents")
    return SplitDefinition(split_hash=expected, **content)
