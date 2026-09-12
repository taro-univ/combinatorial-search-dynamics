"""MLflow local-file recording without a tracking server."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _mlflow():
    # MLflow 3.16+ requires an explicit opt-in for its maintained local file store.
    # These settings contain no credential and are limited to the current process.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    return mlflow


def local_tracking_uri(value: str, *, root: Path) -> str:
    """Resolve the configured local file-store URI without contacting a server."""
    if not value.startswith("file:"):
        raise ValueError("Phase 2 requires an MLflow file: tracking URI")
    location = value.removeprefix("file:")
    path = Path(location)
    if not path.is_absolute():
        path = root / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, name))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            flattened[name] = ",".join(str(part) for part in item)
        else:
            flattened[name] = str(item)
    return flattened


def record_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    artifacts: list[Path],
    prepare_artifacts: Callable[[str], list[Path]] | None = None,
) -> str:
    """Create one completed MLflow run and return its real run ID."""
    try:
        mlflow = _mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as active:
            if prepare_artifacts is not None:
                artifacts = [*artifacts, *prepare_artifacts(active.info.run_id)]
            mlflow.log_params(_flatten(parameters))
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
            return active.info.run_id
    except Exception as exc:
        raise RuntimeError(f"MLflow recording failed: {type(exc).__name__}: {exc}") from exc


def log_artifacts(*, tracking_uri: str, run_id: str, artifacts: list[Path]) -> None:
    """Append report artifacts to an existing local run."""
    try:
        mlflow = _mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception as exc:
        raise RuntimeError(
            f"MLflow artifact recording failed: {type(exc).__name__}: {exc}"
        ) from exc
