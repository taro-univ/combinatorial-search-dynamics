from typer.testing import CliRunner

from llm_search_dynamics.cli import app
from llm_search_dynamics.data.duckdb import create_analysis_views
from llm_search_dynamics.data.validation import validate_dataset
from tests.helpers import write_dataset


def test_phase_one_storage_validation_and_query_flow(tmp_path) -> None:
    write_dataset(tmp_path)

    assert validate_dataset(tmp_path).is_valid
    connection = create_analysis_views(tmp_path)
    assert connection.execute("SELECT count(*) FROM analysis_checkpoints").fetchone()[0] == 3

    result = CliRunner().invoke(app, ["validate-data", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Data validation succeeded" in result.output
