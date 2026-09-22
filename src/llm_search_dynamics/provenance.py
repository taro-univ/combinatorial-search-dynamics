"""Safe, structured provenance collection for local pilot runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_search_dynamics import __version__
from llm_search_dynamics.identifiers import canonical_json


def config_hash(config: dict[str, Any]) -> str:
    """Hash the complete resolved scientific configuration."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _git_value(root: Path, arguments: list[str], fallback: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode:
        return fallback
    return result.stdout.strip()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_provenance(
    *,
    root: Path,
    config: dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, Any]:
    """Collect reproducibility metadata without reading environment variables."""
    lock_path = root / "uv.lock"
    lock_hash = (
        hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else "unavailable"
    )
    status = _git_value(root, ["status", "--porcelain"], "unavailable")
    commit = _git_value(root, ["rev-parse", "HEAD"], "unavailable:not-a-git-checkout")
    return {
        "git_commit": commit,
        "git_dirty": status not in {"", "unavailable"},
        "code_status": "unavailable"
        if status == "unavailable"
        else ("dirty" if status else "clean"),
        "dvc_revision": "unavailable:current-stage-lock-not-finalized",
        "uv_lock_sha256": lock_hash,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine() or "unavailable",
        "package_version": __version__,
        "dependency_versions": {
            "pyarrow": _version("pyarrow"),
            "zarr": _version("zarr"),
            "duckdb": _version("duckdb"),
            "mlflow": _version("mlflow"),
            "dvc": _version("dvc"),
            "ortools": _version("ortools"),
        },
        "unused_phase_2_components": {
            "gpu": "not-used",
            "cuda": "not-used",
            "pytorch": "not-used",
            "transformers": "not-used",
            "ortools": "used-for-references"
            if config["task"]["name"] == "knapsack"
            else "not-used",
        },
        "resolved_config": config,
        "config_hash": config_hash(config),
        "seeds": {
            "instance_generation": config["experiment"]["seed"]["instance_generation"],
            "initial_state": config["experiment"]["seed"]["initial_state"],
            "search": config["experiment"]["seed"]["search"],
            "data_split": config["experiment"]["seed"]["data_split"],
            "representation": config["experiment"]["seed"]["representation"],
            "dynamics": config["experiment"]["seed"]["dynamics"],
        },
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "executable": Path(sys.executable).name,
    }


def write_provenance(path: Path, provenance: dict[str, Any]) -> Path:
    """Write provenance as JSON; callers own collision policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
