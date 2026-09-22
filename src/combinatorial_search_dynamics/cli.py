from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from combinatorial_search_dynamics.config import (
    repository_root,
    resolved_config,
    validate_repository,
)
from combinatorial_search_dynamics.data.parquet import read_parquet
from combinatorial_search_dynamics.data.schemas import TABLE_NAMES
from combinatorial_search_dynamics.data.validation import validate_dataset
from combinatorial_search_dynamics.pipeline.stages import (
    STAGE_NAMES,
    PilotPaths,
    reproduce_pilot,
    run_stage,
)

app = typer.Typer(help="Search Dynamics CLI")


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
    """Check the local development environment and repository configuration."""
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
        typer.echo('Usage: csd codex -- "your prompt"', err=True)
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
def validate_data(
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Phase 1 dataset directory")
    ] = None,
    allow_empty: Annotated[
        bool,
        typer.Option("--allow-empty", help="Return success when no Phase 1 artifacts exist"),
    ] = False,
) -> None:
    """Validate Phase 1 Parquet and Zarr artifacts."""
    if data_dir is None:
        configured = Path(str(resolved_config()["storage"]["data_dir"]))
        data_dir = configured if configured.is_absolute() else repo_root() / configured

    result = validate_dataset(data_dir, allow_empty=allow_empty)
    if result.is_valid:
        target = Path(data_dir)
        has_artifacts = (
            any((target / f"{name}.parquet").exists() for name in TABLE_NAMES)
            or (target / "observations.zarr").exists()
        )
        if allow_empty and not has_artifacts:
            typer.echo(f"No data found; allowed explicitly: {Path(data_dir)}")
        else:
            typer.echo(f"Data validation succeeded: {Path(data_dir)}")
        return

    typer.echo(f"Data validation failed with {result.error_count} error(s):", err=True)
    displayed = result.issues[:50]
    for issue in displayed:
        related = f" id={issue.related_id}" if issue.related_id else ""
        typer.echo(
            f"[{issue.severity.upper()}] {issue.code} ({issue.table_or_artifact}){related}: "
            f"{issue.message}",
            err=True,
        )
    if len(result.issues) > len(displayed):
        typer.echo(f"... {len(result.issues) - len(displayed)} additional issue(s)", err=True)
    raise typer.Exit(code=1)


_PILOT_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


def _run_pilot_command(
    stage: str, ctx: typer.Context, *, force: bool, full_pipeline: bool = False
) -> None:
    """Compose Hydra overrides and run a CPU-only pilot stage."""
    try:
        overrides = list(ctx.args)
        if any("=" not in item for item in overrides):
            raise ValueError("Hydra overrides must be of the form group.key=value")
        config = resolved_config(overrides=overrides)
        if full_pipeline:
            results = reproduce_pilot(config, force=force)
        else:
            results = {stage: run_stage(stage, config, force=force)}
        for completed, outputs in results.items():
            typer.echo(f"{completed}: {', '.join(str(path) for path in outputs)}")
            if completed == "solve-references":
                paths = PilotPaths.from_config(config)
                statuses = Counter(
                    row["solver_status"]
                    for row in read_parquet(paths.reference, "reference_solutions").to_pylist()
                )
                typer.echo(
                    f"  input={paths.raw / 'instances.parquet'}; "
                    + "statuses="
                    + ", ".join(f"{name}:{count}" for name, count in sorted(statuses.items()))
                )
    except Exception as exc:
        # Error artifacts contain only the error class and stage, never env/config values.
        try:
            if "config" in locals():
                path = PilotPaths.from_config(config).artifacts / "errors" / f"{stage}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "stage": stage,
                            "error_type": type(exc).__name__,
                            "recorded_at": datetime.now(UTC).isoformat(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except (OSError, ValueError, KeyError):
            pass
        typer.echo(f"{stage} failed ({type(exc).__name__}): {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _stage_command(stage: str):
    def command(
        ctx: typer.Context,
        force: Annotated[
            bool, typer.Option("--force", help="Replace this stage's outputs")
        ] = False,
    ) -> None:
        _run_pilot_command(stage, ctx, force=force)

    command.__doc__ = f"Run the {stage} CPU pilot stage using Hydra overrides."
    return command


for _stage_name in STAGE_NAMES:
    app.command(name=_stage_name, context_settings=_PILOT_CONTEXT)(_stage_command(_stage_name))


@app.command("reproduce-pilot", context_settings=_PILOT_CONTEXT)
def reproduce_pilot_command(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Replace all pilot outputs")] = False,
) -> None:
    """Run the configured Phase 2 or 3 pilot in-process without an LLM or GPU."""
    _run_pilot_command("reproduce-pilot", ctx, force=force, full_pipeline=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
