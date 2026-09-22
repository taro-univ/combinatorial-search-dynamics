"""Validated Zarr v3 storage for checkpoint-indexed multidimensional observations."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from llm_search_dynamics.data.schemas import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    UnsupportedSchemaVersionError,
)

ZARR_FORMAT = 3
OBSERVATION_GROUPS = ("hidden", "attention_summary", "mlp_update", "kv_summary")
EXTERNAL_OBSERVATION_GROUP = "external_state"
SUPPORTED_OBSERVATION_GROUPS = (*OBSERVATION_GROUPS, EXTERNAL_OBSERVATION_GROUP)


class ZarrValidationError(ValueError):
    """Raised when an observation store is incomplete or structurally invalid."""


@dataclass(frozen=True)
class ObservationArray:
    values: np.ndarray
    valid_mask: np.ndarray
    axis_names: tuple[str, ...]


@dataclass(frozen=True)
class ObservationStore:
    observations: dict[str, ObservationArray]
    trial_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    budget_used: np.ndarray
    search_method_revision: str | None
    legacy_tokenizer_revision: str | None
    observation_code_version: str
    observation_metadata: dict[str, str]

    @property
    def generated_token_indices(self) -> np.ndarray:
        """Legacy alias for schema-v1 readers."""
        return self.budget_used

    @property
    def model_revision(self) -> str | None:
        """Legacy alias retained for schema-v1 callers."""
        return self.search_method_revision

    @property
    def tokenizer_revision(self) -> str | None:
        """Legacy tokenizer metadata has no classical-search equivalent."""
        return self.legacy_tokenizer_revision


def _chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(1, min(length, 64)) for length in shape)


def _validate_inputs(
    observations: dict[str, ObservationArray],
    trial_ids: list[str] | tuple[str, ...],
    checkpoint_ids: list[str] | tuple[str, ...],
    budget_used: list[int] | tuple[int, ...] | np.ndarray,
) -> None:
    if not observations:
        raise ZarrValidationError("At least one observation group is required")
    extra = observations.keys() - set(SUPPORTED_OBSERVATION_GROUPS)
    if extra:
        raise ZarrValidationError(f"Unsupported observation groups: {', '.join(sorted(extra))}")
    if (
        set(observations) != {EXTERNAL_OBSERVATION_GROUP}
        and not set(OBSERVATION_GROUPS) <= observations.keys()
    ):
        raise ZarrValidationError(
            "Internal observations require all four Phase 1 groups; "
            "external-only stores require external_state"
        )

    count = len(trial_ids)
    if len(checkpoint_ids) != count or len(budget_used) != count:
        raise ZarrValidationError("All index arrays must have the same length")
    if any(not isinstance(value, str) or not value for value in trial_ids):
        raise ZarrValidationError("trial_id index values must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in checkpoint_ids):
        raise ZarrValidationError("checkpoint_id index values must be non-empty strings")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
        for value in budget_used
    ):
        raise ZarrValidationError("budget_used values must be non-negative integers")

    for name, observation in observations.items():
        values = np.asarray(observation.values)
        mask = np.asarray(observation.valid_mask)
        if values.ndim == 0:
            raise ZarrValidationError(f"{name} values must have a checkpoint axis")
        if values.shape != mask.shape:
            raise ZarrValidationError(f"{name} valid_mask shape must equal values shape")
        if mask.dtype != np.bool_:
            raise ZarrValidationError(f"{name} valid_mask must have boolean dtype")
        if values.shape[0] != count:
            raise ZarrValidationError(f"{name} first axis must match index length")
        if len(observation.axis_names) != values.ndim:
            raise ZarrValidationError(f"{name} axis_names length must match array rank")
        if not observation.axis_names or observation.axis_names[0] != "checkpoint":
            raise ZarrValidationError(f"{name} first axis must be named checkpoint")


def _create_index_array(group: zarr.Group, name: str, values: list[str]) -> None:
    array = group.create_array(
        name,
        shape=(len(values),),
        dtype="str",
        chunks=(max(1, min(len(values), 1024)),),
        dimension_names=("checkpoint",),
        attributes={"shape": [len(values)], "axis_names": ["checkpoint"], "dtype": "string"},
    )
    array.attrs["dtype"] = str(array.dtype)
    if values:
        array[:] = values


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_observation_store(
    path: Path,
    observations: dict[str, ObservationArray],
    *,
    trial_ids: list[str] | tuple[str, ...],
    checkpoint_ids: list[str] | tuple[str, ...],
    budget_used: list[int] | tuple[int, ...] | np.ndarray,
    observation_code_version: str,
    search_method_revision: str | None = None,
    observation_metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> None:
    """Write a complete Zarr v3 store through a validated temporary directory."""
    _validate_inputs(observations, trial_ids, checkpoint_ids, budget_used)
    if not observation_code_version:
        raise ValueError("observation_code_version must be non-empty")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Zarr store: {destination}")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.backup"
    try:
        root = zarr.open_group(temporary, mode="w", zarr_format=ZARR_FORMAT)
        root.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "zarr_format": ZARR_FORMAT,
                "status": "writing",
                "search_method_revision": search_method_revision,
                "observation_code_version": observation_code_version,
                "observation_metadata": observation_metadata or {},
            }
        )

        for name, observation in observations.items():
            values = np.asarray(observation.values)
            mask = np.asarray(observation.valid_mask)
            group = root.create_group(name)
            common = {"shape": list(values.shape), "axis_names": list(observation.axis_names)}
            group.create_array(
                "values",
                data=values,
                chunks=_chunks(values.shape),
                dimension_names=observation.axis_names,
                attributes={**common, "dtype": str(values.dtype)},
            )
            group.create_array(
                "valid_mask",
                data=mask,
                chunks=_chunks(mask.shape),
                dimension_names=observation.axis_names,
                attributes={**common, "dtype": str(mask.dtype)},
            )

        index = root.create_group("index")
        _create_index_array(index, "trial_id", list(trial_ids))
        _create_index_array(index, "checkpoint_id", list(checkpoint_ids))
        positions_array = np.asarray(budget_used, dtype=np.int64)
        index.create_array(
            "budget_used",
            data=positions_array,
            chunks=(max(1, min(len(positions_array), 1024)),),
            dimension_names=("checkpoint",),
            attributes={
                "shape": [len(positions_array)],
                "axis_names": ["checkpoint"],
                "dtype": "int64",
            },
        )
        root.attrs["status"] = "complete"
        read_observation_store(temporary)

        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing Zarr store: {destination}")
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except OSError:
            if backup.exists():
                os.replace(backup, destination)
            raise
        _remove_path(backup)
    finally:
        _remove_path(temporary)
        if backup.exists():
            if destination.exists():
                _remove_path(backup)
            else:
                os.replace(backup, destination)


def _array_metadata_issues(array: zarr.Array, expected_axis_names: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    attrs = dict(array.attrs)
    if attrs.get("shape") != list(array.shape):
        issues.append("recorded shape does not match array shape")
    if attrs.get("axis_names") != list(expected_axis_names):
        issues.append("recorded axis_names do not match")
    if attrs.get("dtype") != str(array.dtype):
        issues.append("recorded dtype does not match array dtype")
    return issues


def read_observation_store(path: Path) -> ObservationStore:
    """Open a complete Zarr v3 store and validate version, structure, and indexes."""
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"Zarr store does not exist: {source}")
    root = zarr.open_group(source, mode="r")
    attrs = dict(root.attrs)
    if root.metadata.zarr_format != ZARR_FORMAT:
        raise ZarrValidationError(f"Expected Zarr format {ZARR_FORMAT}")
    version = str(attrs.get("schema_version", ""))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(
            f"Unsupported Zarr schema version: {attrs.get('schema_version', '<missing>')}"
        )
    if attrs.get("zarr_format") != ZARR_FORMAT:
        raise ZarrValidationError(f"Expected Zarr format {ZARR_FORMAT}")
    if attrs.get("status") != "complete":
        raise ZarrValidationError("Zarr store is not marked complete")
    if "index" not in root:
        raise ZarrValidationError("Zarr index group is missing")

    index = root["index"]
    position_name = "generated_token_index" if version == "1" else "budget_used"
    required_indexes = ("trial_id", "checkpoint_id", position_name)
    missing_indexes = [name for name in required_indexes if name not in index]
    if missing_indexes:
        raise ZarrValidationError(f"Missing Zarr indexes: {', '.join(missing_indexes)}")

    index_issues: list[str] = []
    for name in required_indexes:
        index_issues.extend(
            f"index.{name} {issue}"
            for issue in _array_metadata_issues(index[name], ("checkpoint",))
        )
    if index_issues:
        raise ZarrValidationError("; ".join(index_issues))

    trial_ids = tuple(str(value) for value in index["trial_id"][:].tolist())
    checkpoint_ids = tuple(str(value) for value in index["checkpoint_id"][:].tolist())
    positions = np.asarray(index[position_name][:])
    count = len(trial_ids)
    if len(checkpoint_ids) != count or len(positions) != count:
        raise ZarrValidationError("Zarr index arrays have different lengths")

    observations: dict[str, ObservationArray] = {}
    structural_issues: list[str] = []
    observation_names = sorted(name for name in root.group_keys() if name != "index")
    if not observation_names:
        structural_issues.append("No observation groups are present")
    unsupported = set(observation_names) - set(SUPPORTED_OBSERVATION_GROUPS)
    if unsupported:
        structural_issues.append(
            f"Unsupported observation groups: {', '.join(sorted(unsupported))}"
        )
    for name in observation_names:
        group = root[name]
        if "values" not in group or "valid_mask" not in group:
            structural_issues.append(f"{name} must contain values and valid_mask")
            continue
        values_array = group["values"]
        mask_array = group["valid_mask"]
        values = np.asarray(values_array[:])
        mask = np.asarray(mask_array[:])
        axis_names = tuple(values_array.attrs.get("axis_names", ()))
        structural_issues.extend(
            f"{name}.values {issue}" for issue in _array_metadata_issues(values_array, axis_names)
        )
        structural_issues.extend(
            f"{name}.valid_mask {issue}" for issue in _array_metadata_issues(mask_array, axis_names)
        )
        observations[name] = ObservationArray(values, mask, axis_names)

    if structural_issues:
        raise ZarrValidationError("; ".join(structural_issues))
    _validate_inputs(observations, trial_ids, checkpoint_ids, positions)
    if not attrs.get("observation_code_version"):
        raise ZarrValidationError("observation_code_version root attribute is missing")
    return ObservationStore(
        observations=observations,
        trial_ids=trial_ids,
        checkpoint_ids=checkpoint_ids,
        budget_used=positions,
        search_method_revision=attrs.get("search_method_revision") or attrs.get("model_revision"),
        legacy_tokenizer_revision=attrs.get("tokenizer_revision"),
        observation_code_version=str(attrs.get("observation_code_version", "")),
        observation_metadata={
            str(key): str(value) for key, value in attrs.get("observation_metadata", {}).items()
        },
    )
