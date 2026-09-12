from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer

from llm_search_dynamics.config import repository_root, validate_repository

app = typer.Typer(help="LLM Search Dynamics CLI")


def repo_root() -> Path:
    """Return the repository root (compatibility wrapper)."""
    return repository_root()


def read_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without mutating the process environment."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            env[key] = value
    return env


@app.command()
def doctor() -> None:
    """Check the local Phase 0 development environment."""
    py_version = sys.version.split()[0]
    repository_errors = validate_repository(repo_root())
    checks = [
        ("Python 3.11 runtime", sys.version_info[:2] == (3, 11), py_version),
        ("uv installed", shutil.which("uv") is not None, shutil.which("uv") or "missing"),
        (
            "repository and Hydra configuration",
            not repository_errors,
            "valid" if not repository_errors else "invalid",
        ),
    ]

    for name, ok, detail in checks:
        label = "OK" if ok else "MISSING"
        typer.echo(f"[{label}] {name}: {detail}")

    if repository_errors:
        for error in repository_errors:
            typer.echo(f"  - {error}", err=True)
    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(code=1)


@app.command()
def bootstrap() -> None:
    """Run the repository bootstrap flow."""
    script = repo_root() / "scripts" / "bootstrap.sh"
    if not script.exists():
        typer.echo("Missing bootstrap script.", err=True)
        raise typer.Exit(code=1)
    result = subprocess.run(["bash", str(script)], check=False)
    raise typer.Exit(code=result.returncode)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def codex(ctx: typer.Context) -> None:
    """Run the separately installed Codex development CLI."""
    prompt = " ".join(ctx.args)
    if not prompt:
        typer.echo('Usage: lsd codex -- "your prompt"', err=True)
        raise typer.Exit(code=1)

    result = subprocess.run(["codex", "run", prompt], check=False)
    raise typer.Exit(code=result.returncode)


def _run_repository_validation() -> None:
    errors = validate_repository(repo_root())
    if errors:
        typer.echo("Repository validation failed:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Repository structure and Hydra configuration are valid.")


@app.command("validate-repository")
def validate_repository_command() -> None:
    """Validate the Phase 0 repository structure and Hydra configuration."""
    _run_repository_validation()


@app.command("validate-data")
def validate_data() -> None:
    """Compatibility alias; Phase 0 has no experiment data to validate."""
    typer.echo(
        "Phase 0 compatibility alias: validating the repository; no data validation is implemented."
    )
    _run_repository_validation()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
