import numpy as np
import pytest
import zarr

from llm_search_dynamics.data.schemas import UnsupportedSchemaVersionError
from llm_search_dynamics.data.zarr import (
    OBSERVATION_GROUPS,
    ObservationArray,
    ZarrValidationError,
    read_observation_store,
    write_observation_store,
)
from tests.helpers import make_observations, make_tables


def _write(path, observations=None, trial_ids=None) -> None:
    tables = make_tables()
    checkpoints = tables["checkpoints"]
    write_observation_store(
        path,
        observations or make_observations(tables),
        trial_ids=trial_ids or checkpoints.column("trial_id").to_pylist(),
        checkpoint_ids=checkpoints.column("checkpoint_id").to_pylist(),
        generated_token_indices=checkpoints.column("generated_token_index").to_pylist(),
        observation_code_version="test-v1",
        model_revision="model-r1",
        tokenizer_revision="tokenizer-r1",
    )


def test_observations_masks_indexes_and_attributes_round_trip(tmp_path) -> None:
    path = tmp_path / "observations.zarr"
    _write(path)
    stored = read_observation_store(path)
    assert set(stored.observations) == set(OBSERVATION_GROUPS)
    assert stored.model_revision == "model-r1"
    assert stored.tokenizer_revision == "tokenizer-r1"
    assert stored.observation_code_version == "test-v1"
    root = zarr.open_group(path, mode="r")
    assert root.metadata.zarr_format == 3
    assert root["hidden"]["values"].attrs["axis_names"] == ["checkpoint", "feature"]
    assert root["hidden"]["values"].attrs["dtype"] == "float32"
    np.testing.assert_array_equal(
        stored.observations["hidden"].valid_mask,
        np.ones((3, 2), dtype=np.bool_),
    )


def test_mask_shape_mismatch_is_rejected(tmp_path) -> None:
    tables = make_tables()
    observations = make_observations(tables)
    observations["hidden"] = ObservationArray(
        observations["hidden"].values,
        np.ones((3, 1), dtype=np.bool_),
        ("checkpoint", "feature"),
    )
    with pytest.raises(ZarrValidationError, match="valid_mask shape"):
        _write(tmp_path / "observations.zarr", observations=observations)


def test_index_length_mismatch_is_rejected(tmp_path) -> None:
    with pytest.raises(ZarrValidationError, match="index arrays"):
        _write(tmp_path / "observations.zarr", trial_ids=["trl_only_one"])


def test_version_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "observations.zarr"
    _write(path)
    zarr.open_group(path, mode="a").attrs["schema_version"] = "999"
    with pytest.raises(UnsupportedSchemaVersionError):
        read_observation_store(path)


def test_incomplete_store_is_rejected(tmp_path) -> None:
    path = tmp_path / "observations.zarr"
    _write(path)
    zarr.open_group(path, mode="a").attrs["status"] = "writing"
    with pytest.raises(ZarrValidationError, match="not marked complete"):
        read_observation_store(path)


def test_implicit_store_overwrite_is_rejected(tmp_path) -> None:
    path = tmp_path / "observations.zarr"
    _write(path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write(path)


def test_explicit_store_overwrite_replaces_complete_store(tmp_path) -> None:
    path = tmp_path / "observations.zarr"
    _write(path)
    tables = make_tables()
    checkpoints = tables["checkpoints"]
    write_observation_store(
        path,
        make_observations(tables),
        trial_ids=checkpoints.column("trial_id").to_pylist(),
        checkpoint_ids=checkpoints.column("checkpoint_id").to_pylist(),
        generated_token_indices=checkpoints.column("generated_token_index").to_pylist(),
        observation_code_version="replacement-v2",
        overwrite=True,
    )
    assert read_observation_store(path).observation_code_version == "replacement-v2"
