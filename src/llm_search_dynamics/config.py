"""Hydra configuration loading and repository validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

CONFIG_GROUPS = (
    "experiment",
    "task",
    "llm",
    "generation",
    "observation",
    "state_model",
    "dynamics",
    "evaluation",
    "storage",
    "tracking",
)


def repository_root() -> Path:
    """Return the repository root for this src-layout package."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Could not locate repository root from {Path(__file__).resolve()}")
    return root


def compose_config(
    *,
    config_dir: Path | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra configuration, failing clearly when its root file is absent."""
    directory = (config_dir or repository_root() / "configs").resolve()
    config_file = directory / f"{config_name}.yaml"
    if not directory.is_dir():
        raise FileNotFoundError(f"Hydra config directory does not exist: {directory}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Hydra config file does not exist: {config_file}")

    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


def resolved_config(**kwargs: Any) -> dict[str, Any]:
    """Return the fully resolved Hydra configuration as plain Python data."""
    resolved = OmegaConf.to_container(compose_config(**kwargs), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("The composed Hydra configuration must be a mapping")
    return resolved


def validate_repository(root: Path | None = None) -> list[str]:
    """Return repository validation errors; an empty list means success."""
    target = (root or repository_root()).resolve()
    required_files = (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".env.example",
        "docs/repository-specification.md",
        "docs/architecture.md",
        "docs/experiment-protocol.md",
        "docs/data-dictionary.md",
        "docs/adr/README.md",
        "docs/adr/ADR-001-uv-and-python-311.md",
        "docs/adr/ADR-002-src-layout.md",
        "docs/adr/ADR-003-hydra-as-config-source.md",
        "docs/adr/ADR-004-parquet-and-zarr-responsibilities.md",
        "docs/adr/ADR-005-duckdb-query-layer.md",
        "docs/adr/ADR-006-git-dvc-mlflow-responsibilities.md",
        "dvc.yaml",
        ".dvcignore",
        "scripts/bootstrap.sh",
        ".github/workflows/ci.yml",
        "src/llm_search_dynamics/__init__.py",
        "src/llm_search_dynamics/cli.py",
        "src/llm_search_dynamics/config.py",
        "src/llm_search_dynamics/identifiers.py",
        "src/llm_search_dynamics/data/__init__.py",
        "src/llm_search_dynamics/data/schemas.py",
        "src/llm_search_dynamics/data/parquet.py",
        "src/llm_search_dynamics/data/zarr.py",
        "src/llm_search_dynamics/data/duckdb.py",
        "src/llm_search_dynamics/data/validation.py",
        "configs/config.yaml",
        "configs/experiment/pilot.yaml",
        "configs/task/placeholder.yaml",
        "configs/llm/placeholder.yaml",
        "configs/generation/default.yaml",
        "configs/observation/external_only.yaml",
        "configs/state_model/baseline.yaml",
        "configs/dynamics/markov.yaml",
        "configs/evaluation/default.yaml",
        "configs/storage/local.yaml",
        "configs/tracking/local.yaml",
        "configs/task/dummy_binary.yaml",
        "configs/llm/mock.yaml",
        "configs/generation/mock.yaml",
    )
    errors = [
        f"missing required file: {path}" for path in required_files if not (target / path).is_file()
    ]
    python_version_file = target / ".python-version"
    if (
        python_version_file.is_file()
        and python_version_file.read_text(encoding="utf-8").strip() != "3.11"
    ):
        errors.append(".python-version must specify Python 3.11")

    try:
        config = resolved_config(config_dir=target / "configs")
    except (FileNotFoundError, HydraException, OmegaConfBaseException, TypeError) as exc:
        errors.append(f"Hydra composition failed: {exc}")
        return errors

    missing_groups = [group for group in CONFIG_GROUPS if group not in config]
    if missing_groups:
        errors.append("resolved Hydra config is missing groups: " + ", ".join(missing_groups))
    return errors
