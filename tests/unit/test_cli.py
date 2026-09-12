from pathlib import Path

from typer.testing import CliRunner

from llm_search_dynamics.cli import app, read_env, repo_root


def test_repository_root_is_found() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "docs" / "repository-specification.md").is_file()


def test_read_env_handles_blank_comments_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# comment\nPLAIN=value\nDOUBLE=\"two words\"\nSINGLE='three words'\n",
        encoding="utf-8",
    )

    assert read_env(env_file) == {
        "PLAIN": "value",
        "DOUBLE": "two words",
        "SINGLE": "three words",
    }


def test_doctor_does_not_print_secret_values(monkeypatch) -> None:
    secret = "phase-zero-secret-must-not-appear"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert secret not in result.output
