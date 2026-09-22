"""Deterministic, content-derived identifiers.

Values are encoded as UTF-8 canonical JSON with sorted object keys, compact
separators, preserved Unicode, and finite JSON numbers only. IDs use SHA-256,
truncated to 24 hexadecimal characters (96 bits), plus a type prefix. The
canonicalizer accepts JSON scalars, lists, and string-keyed nested mappings;
NaN, infinity, non-string keys, and non-JSON objects are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

HASH_HEX_LENGTH = 24


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float is not allowed at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings at {path}")
            _validate_json(item, f"{path}.{key}")
        return
    raise TypeError(f"Value at {path} is not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used by every ID."""
    _validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:HASH_HEX_LENGTH]}"


def instance_id(instance: Mapping[str, Any]) -> str:
    """Return an ID derived only from a canonical problem definition."""
    return _content_id("ins", {"instance": instance})


def experiment_id(name: str, conditions: Mapping[str, Any]) -> str:
    """Return a readable experiment name plus a deterministic condition hash."""
    if not name or not isinstance(name, str):
        raise ValueError("Experiment name must be a non-empty string")
    return f"exp_{name}_{_content_id('cfg', conditions).removeprefix('cfg_')}"


def trial_id(
    source_instance_id: str,
    experiment_conditions: Mapping[str, Any],
    search_seed: int,
) -> str:
    """Derive a trial ID from instance, complete conditions, and search seed."""
    if not source_instance_id:
        raise ValueError("source_instance_id must be non-empty")
    if isinstance(search_seed, bool) or not isinstance(search_seed, int):
        raise TypeError("search_seed must be an integer")
    return _content_id(
        "trl",
        {
            "instance_id": source_instance_id,
            "experiment_conditions": experiment_conditions,
            "search_seed": search_seed,
        },
    )


def checkpoint_id(source_trial_id: str, checkpoint_index: int) -> str:
    """Derive a checkpoint ID from a trial ID and non-negative checkpoint position."""
    if not source_trial_id:
        raise ValueError("source_trial_id must be non-empty")
    if (
        isinstance(checkpoint_index, bool)
        or not isinstance(checkpoint_index, int)
        or checkpoint_index < 0
    ):
        raise ValueError("checkpoint_index must be a non-negative integer")
    return _content_id(
        "chk",
        {"trial_id": source_trial_id, "checkpoint_index": checkpoint_index},
    )


def artifact_id(content_descriptor: Mapping[str, Any]) -> str:
    """Derive an artifact ID from a stable, JSON-compatible content descriptor."""
    return _content_id("art", {"content": content_descriptor})
