import pytest

from llm_search_dynamics.data.duckdb import MissingParquetError, create_analysis_views
from tests.helpers import write_dataset


def test_analysis_checkpoints_joins_multiple_trials_by_id(tmp_path) -> None:
    tables = write_dataset(tmp_path)
    connection = create_analysis_views(tmp_path)
    rows = connection.execute(
        "SELECT instance_id, trial_id, checkpoint_id FROM analysis_checkpoints "
        "ORDER BY trial_id, checkpoint_index"
    ).fetchall()
    assert len(rows) == 3
    assert {row[0] for row in rows} == set(tables["instances"].column("instance_id").to_pylist())
    assert len({row[1] for row in rows}) == 2
    assert len({row[2] for row in rows}) == 3
    assert connection.execute("SELECT count(*) FROM metrics").fetchone()[0] == 2


def test_missing_parquet_fails_clearly(tmp_path) -> None:
    with pytest.raises(MissingParquetError, match="instances.parquet"):
        create_analysis_views(tmp_path)
