from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from zhou_aim.config import load_config, profile_path
from zhou_aim.dynamics import SimulationConfig, SimulationResult
from zhou_aim.provenance import collect_provenance, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def load_profile(name: str) -> dict[str, Any]:
    return load_config(profile_path(PROJECT_ROOT, name))


def simulation_config(config: dict[str, Any], *, model_kind: str | None = None) -> SimulationConfig:
    model = config["model"]
    integration = config["integration"]
    return SimulationConfig(
        model_kind=model_kind or str(model["kind"]),
        dt=float(integration["dt"]),
        steps=int(integration["steps"]),
        record_stride=int(integration["record_stride"]),
        xi=float(model["xi"]),
        initial_std=float(integration["initial_std"]),
        initialization=str(integration["initialization"]),
        boundary=float(model["boundary"]),
        project_after_step=bool(model["project_after_step"]),
        flip_window_steps=int(integration["flip_window_steps"]),
        kde_bandwidth_factor=float(integration["kde_bandwidth_factor"]),
        p0_bin_width=float(integration["p0_bin_width"]),
    )


def unique_run_dir(category: str, run_id: str | None) -> Path:
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = PROJECT_ROOT / "results" / category / identifier
    path.mkdir(parents=True, exist_ok=False)
    return path


def result_frame(result: SimulationResult, **labels: object) -> pd.DataFrame:
    scalar_fields = [
        "time",
        "step",
        "energy",
        "best_energy",
        "residual",
        "best_residual",
        "p0_kde",
        "p0_hist",
        "kde_bandwidth",
        "flip_rate",
        "sqrt_dt_mu",
        "p0_mu_prediction",
        "cumulative_flips",
        "amplitude_mean",
        "amplitude_variance",
        "overlap_initial",
        "overlap_best",
        "soft_fraction",
        "gain",
        "temperature",
    ]
    frame = pd.DataFrame({name: getattr(result, name) for name in scalar_fields})
    for name, value in labels.items():
        frame[name] = value
    return frame


def save_manifest(run_dir: Path, config: dict[str, Any], details: dict[str, Any]) -> None:
    payload = {
        "provenance": collect_provenance(REPOSITORY_ROOT),
        "config": config,
        "details": details,
    }
    write_json(run_dir / "manifest.json", payload)


def print_summary(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def config_dict(config: SimulationConfig) -> dict[str, Any]:
    return asdict(config)
