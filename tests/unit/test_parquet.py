import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from combinatorial_search_dynamics.data.parquet import read_parquet, write_parquet
from combinatorial_search_dynamics.data.schemas import (
    TABLE_NAMES,
    SchemaValidationError,
    UnsupportedSchemaVersionError,
    empty_table,
)
from tests.helpers import make_tables


@pytest.mark.parametrize("table_name", TABLE_NAMES)
def test_table_round_trip_uses_zstandard(tmp_path, table_name: str) -> None:
    path = tmp_path / f"{table_name}.parquet"
    table = make_tables()[table_name]
    write_parquet(table, path, table_name)
    assert read_parquet(path, table_name).equals(table)
    metadata = pq.ParquetFile(path).metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("table_name", TABLE_NAMES)
def test_empty_table_round_trip(tmp_path, table_name: str) -> None:
    path = tmp_path / f"empty-{table_name}.parquet"
    write_parquet(empty_table(table_name), path, table_name)
    assert read_parquet(path, table_name).num_rows == 0


def test_implicit_overwrite_is_rejected(tmp_path) -> None:
    table = make_tables()["instances"]
    path = tmp_path / "instances.parquet"
    write_parquet(table, path, "instances")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_parquet(table, path, "instances")


def test_explicit_overwrite_replaces_file(tmp_path) -> None:
    table = make_tables()["instances"]
    path = tmp_path / "instances.parquet"
    write_parquet(table, path, "instances")
    write_parquet(table, path, "instances", overwrite=True)
    assert read_parquet(path, "instances").equals(table)


def test_version_mismatch_is_rejected_on_read(tmp_path) -> None:
    table = make_tables()["instances"]
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = b"1"
    path = tmp_path / "instances.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(UnsupportedSchemaVersionError):
        read_parquet(path, "instances")


def test_schema_v1_can_be_read_but_not_written(tmp_path) -> None:
    current = make_tables()["instances"]
    rows = [{**row, "schema_version": "1"} for row in current.to_pylist()]
    from combinatorial_search_dynamics.data.schemas import get_schema

    legacy = pa.Table.from_pylist(rows, schema=get_schema("instances", "1"))
    path = tmp_path / "legacy.parquet"
    pq.write_table(legacy, path)
    assert read_parquet(path, "instances").equals(legacy)
    with pytest.raises(ValueError, match="read-only"):
        write_parquet(legacy, tmp_path / "legacy-copy.parquet", "instances")


def test_invalid_schema_is_rejected_before_write(tmp_path) -> None:
    table = make_tables()["instances"].append_column("extra", pa.array([1]))
    with pytest.raises(SchemaValidationError):
        write_parquet(table, tmp_path / "instances.parquet", "instances")
    assert not (tmp_path / "instances.parquet").exists()
