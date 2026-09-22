"""Strict, non-destructive Parquet I/O for canonical logical tables."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from combinatorial_search_dynamics.data.schemas import (
    SCHEMA_VERSION,
    table_schema_version,
    validate_table_schema,
)

DEFAULT_COMPRESSION = "zstd"


def read_parquet(path: Path, table_name: str) -> pa.Table:
    """Read a Parquet table and enforce its schema and supported version."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {source}")
    table = pq.read_table(source)
    validate_table_schema(table, table_name)
    return table


def write_parquet(
    table: pa.Table,
    path: Path,
    table_name: str,
    *,
    overwrite: bool = False,
    compression: str = DEFAULT_COMPRESSION,
) -> None:
    """Validate and atomically write a table, refusing implicit replacement.

    Data is written and read back from a uniquely named temporary file in the
    destination directory. Only a validated temporary file is atomically renamed
    to the final path. Raw callers therefore receive append-only behavior unless
    they explicitly opt into ``overwrite=True``.
    """
    validate_table_schema(table, table_name)
    if table_schema_version(table) != SCHEMA_VERSION:
        raise ValueError(
            "Legacy schemas are read-only; new Parquet writes must use schema version 2"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Parquet file: {destination}")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        pq.write_table(table, temporary, compression=compression)
        read_parquet(temporary, table_name)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Parquet file: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
