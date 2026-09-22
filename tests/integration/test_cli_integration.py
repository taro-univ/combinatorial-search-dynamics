from typer.testing import CliRunner

from combinatorial_search_dynamics.cli import app


def test_validate_repository_needs_no_external_services() -> None:
    result = CliRunner().invoke(app, ["validate-repository"])

    assert result.exit_code == 0, result.output
    assert "Hydra configuration are valid" in result.output
