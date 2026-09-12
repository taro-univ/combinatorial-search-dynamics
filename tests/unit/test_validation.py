import pyarrow as pa
import pyarrow.parquet as pq

from llm_search_dynamics.data.parquet import write_parquet
from llm_search_dynamics.data.validation import validate_dataset
from llm_search_dynamics.data.zarr import write_observation_store
from tests.helpers import make_observations, make_tables, write_dataset


def _codes(path) -> set[str]:
    return {issue.code for issue in validate_dataset(path).issues}


def test_valid_dataset_has_no_errors(tmp_path) -> None:
    write_dataset(tmp_path)
    result = validate_dataset(tmp_path)
    assert result.is_valid
    assert result.issues == ()


def test_primary_key_duplicate_is_detected(tmp_path) -> None:
    tables = make_tables()
    tables["instances"] = pa.concat_tables([tables["instances"], tables["instances"]])
    write_dataset(tmp_path, tables)
    assert "primary_key_duplicate" in _codes(tmp_path)


def test_foreign_key_violation_is_detected(tmp_path) -> None:
    tables = make_tables()
    trials = tables["trials"]
    tables["trials"] = trials.set_column(
        trials.schema.get_field_index("instance_id"),
        trials.schema.field("instance_id"),
        pa.array(["ins_missing", "ins_missing"], type=pa.string()),
    )
    write_dataset(tmp_path, tables)
    assert "foreign_key" in _codes(tmp_path)


def test_non_null_violation_is_detected(tmp_path) -> None:
    tables = make_tables()
    instances = tables["instances"]
    tables["instances"] = instances.set_column(
        instances.schema.get_field_index("instance_id"),
        pa.field("instance_id", pa.string(), nullable=True),
        pa.array([None], type=pa.string()),
    )
    for name, table in tables.items():
        if name == "instances":
            pq.write_table(table, tmp_path / f"{name}.parquet")
        else:
            write_parquet(table, tmp_path / f"{name}.parquet", name)
    assert {"null_value", "primary_key_null"} <= _codes(tmp_path)


def test_checkpoint_order_violation_is_detected(tmp_path) -> None:
    tables = make_tables()
    checkpoints = tables["checkpoints"]
    tables["checkpoints"] = checkpoints.set_column(
        checkpoints.schema.get_field_index("generated_token_index"),
        checkpoints.schema.field("generated_token_index"),
        pa.array([8, 0, 0], type=pa.int64()),
    )
    write_dataset(tmp_path, tables)
    assert "checkpoint_order" in _codes(tmp_path)


def test_duplicate_checkpoint_index_within_trial_is_detected(tmp_path) -> None:
    tables = make_tables()
    checkpoints = tables["checkpoints"]
    tables["checkpoints"] = checkpoints.set_column(
        checkpoints.schema.get_field_index("checkpoint_index"),
        checkpoints.schema.field("checkpoint_index"),
        pa.array([0, 0, 0], type=pa.int64()),
    )
    write_dataset(tmp_path, tables)
    assert "checkpoint_index_duplicate" in _codes(tmp_path)


def test_zarr_checkpoint_mismatch_is_detected(tmp_path) -> None:
    tables = make_tables()
    for name, table in tables.items():
        write_parquet(table, tmp_path / f"{name}.parquet", name)
    checkpoints = tables["checkpoints"]
    zarr_ids = checkpoints.column("checkpoint_id").to_pylist()
    zarr_ids[0] = "chk_unknown"
    write_observation_store(
        tmp_path / "observations.zarr",
        make_observations(tables),
        trial_ids=checkpoints.column("trial_id").to_pylist(),
        checkpoint_ids=zarr_ids,
        generated_token_indices=checkpoints.column("generated_token_index").to_pylist(),
        observation_code_version="test-v1",
    )
    assert {"zarr_checkpoint_missing", "zarr_checkpoint_unknown"} <= _codes(tmp_path)


def test_empty_dataset_requires_explicit_permission(tmp_path) -> None:
    assert not validate_dataset(tmp_path).is_valid
    assert validate_dataset(tmp_path, allow_empty=True).is_valid


def test_incomplete_artifact_is_not_treated_as_empty(tmp_path) -> None:
    (tmp_path / ".instances.parquet.partial").write_bytes(b"incomplete")
    result = validate_dataset(tmp_path, allow_empty=True)
    assert not result.is_valid
    assert "incomplete_artifact" in {issue.code for issue in result.issues}


def test_invalid_split_status_and_success_are_detected(tmp_path) -> None:
    tables = make_tables()
    metrics = tables["metrics"]
    tables["metrics"] = metrics.set_column(
        metrics.schema.get_field_index("split"),
        metrics.schema.field("split"),
        pa.array(["development", "test"], type=pa.string()),
    )
    trials = tables["trials"]
    tables["trials"] = trials.set_column(
        trials.schema.get_field_index("status"),
        trials.schema.field("status"),
        pa.array(["failed", "unknown"], type=pa.string()),
    )
    write_dataset(tmp_path, tables)
    assert {"invalid_split", "invalid_status", "success_terminal_contradiction"} <= _codes(tmp_path)
