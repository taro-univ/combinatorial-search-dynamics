import pyarrow as pa
import pytest

from llm_search_dynamics.data.schemas import (
    SCHEMA_VERSION,
    TABLE_NAMES,
    UnsupportedSchemaVersionError,
    get_schema,
    schema_issues,
    validate_table_schema,
)
from tests.helpers import make_tables


def test_all_four_schemas_are_registered() -> None:
    assert TABLE_NAMES == ("instances", "trials", "checkpoints", "metrics")
    assert all(get_schema(name).field("schema_version").nullable is False for name in TABLE_NAMES)
    assert get_schema("instances").field("created_at").type == pa.timestamp("us", tz="UTC")


def test_schema_metadata_contains_table_and_version() -> None:
    for name in TABLE_NAMES:
        metadata = get_schema(name).metadata
        assert metadata[b"logical_table"] == name.encode()
        assert metadata[b"schema_version"] == SCHEMA_VERSION.encode()


def test_schema_issues_detect_missing_extra_and_type() -> None:
    table = make_tables()["instances"].drop(["task_name"])
    table = table.append_column("extra", pa.array([1], type=pa.int64()))
    problem_index = table.schema.get_field_index("problem_size")
    table = table.set_column(
        problem_index,
        pa.field("problem_size", pa.string(), nullable=False),
        pa.array(["two"]),
    )
    codes = {issue.code for issue in schema_issues(table, "instances")}
    assert {"missing_columns", "extra_columns", "column_type"} <= codes


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown logical table"):
        get_schema("unknown")


def test_unsupported_schema_version_is_rejected() -> None:
    table = make_tables()["instances"]
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = b"999"
    table = table.replace_schema_metadata(metadata)
    with pytest.raises(UnsupportedSchemaVersionError):
        validate_table_schema(table, "instances")
