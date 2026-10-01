from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .observables import (
    decoded_energy_density,
    decoded_spins,
    p0_gaussian_kde,
    p0_histogram,
)
from .schedules import Schedule
from .sk_instance import SKInstance


@dataclass(frozen=True)
class SimulationConfig:
    model_kind: str = "cim"
    dt: float = 0.02
    steps: int = 2000
    record_stride: int = 10
    xi: float = 1.0
    initial_std: float = 0.1
    initialization: str = "gaussian"
    boundary: float = 1.0
    project_after_step: bool = True
    flip_window_steps: int = 10
    kde_bandwidth_factor: float = 1.0
    p0_bin_width: float = 0.1

    def validate(self) -> None:
        if self.model_kind not in {"cim", "digcim", "simcim"}:
            raise ValueError(f"Unknown model_kind: {self.model_kind}")
        if self.dt <= 0 or self.steps < 1 or self.record_stride < 1:
            raise ValueError("dt, steps, and record_stride must be positive")
        if self.boundary <= 0 or self.flip_window_steps < 1:
            raise ValueError("boundary and flip_window_steps must be positive")


@dataclass
class SimulationResult:
    time: np.ndarray
    step: np.ndarray
    x: np.ndarray
    spins: np.ndarray
    energy: np.ndarray
    best_energy: np.ndarray
    residual: np.ndarray
    best_residual: np.ndarray
    p0_kde: np.ndarray
    p0_hist: np.ndarray
    kde_bandwidth: np.ndarray
    flip_rate: np.ndarray
    sqrt_dt_mu: np.ndarray
    p0_mu_prediction: np.ndarray
    cumulative_flips: np.ndarray
    amplitude_mean: np.ndarray
    amplitude_variance: np.ndarray
    overlap_initial: np.ndarray
    overlap_best: np.ndarray
    soft_fraction: np.ndarray
    gain: np.ndarray
    temperature: np.ndarray
    reference_energy: float
    config: SimulationConfig
    seed: int
    schedule_name: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: value for name, value in vars(self).items() if isinstance(value, np.ndarray)
        }
        metadata = {
            "reference_energy": self.reference_energy,
            "config": asdict(self.config),
            "seed": self.seed,
            "schedule_name": self.schedule_name,
        }
        np.savez_compressed(
            path, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True))
        )

    @classmethod
    def load(cls, path: str | Path) -> SimulationResult:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            arrays = {
                name: np.asarray(payload[name]) for name in payload.files if name != "metadata"
            }
        return cls(
            **arrays,
            reference_energy=float(metadata["reference_energy"]),
            config=SimulationConfig(**metadata["config"]),
            seed=int(metadata["seed"]),
            schedule_name=str(metadata["schedule_name"]),
        )


def _initial_state(config: SimulationConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    if config.initialization == "gaussian":
        return rng.normal(0.0, config.initial_std, size=n)
    if config.initialization == "uniform":
        return rng.uniform(-config.initial_std, config.initial_std, size=n)
    if config.initialization == "zeros":
        return np.zeros(n, dtype=np.float64)
    raise ValueError(f"Unknown initialization: {config.initialization}")


def _drift(
    x: np.ndarray,
    spins: np.ndarray,
    couplings: np.ndarray,
    gain: float,
    config: SimulationConfig,
) -> np.ndarray:
    if config.model_kind == "cim":
        return -(x**3) + gain * x + config.xi * (couplings @ x)
    clipped = np.clip(x, -config.boundary, config.boundary)
    if config.model_kind == "simcim":
        return gain * clipped + config.xi * (couplings @ clipped)
    return gain * clipped + config.xi * (couplings @ spins)


def simulate(
    instance: SKInstance,
    schedule: Schedule,
    config: SimulationConfig,
    seed: int,
    reference_energy: float,
    *,
    initial_state: np.ndarray | None = None,
    noise_normals: np.ndarray | None = None,
    external_fields: Mapping[int, np.ndarray] | None = None,
) -> SimulationResult:
    """Euler–Maruyama simulation of a direct finite-N AIM.

    ``external_fields[step]`` is an h-vector present in the drift for exactly one integration step.
    Supplying ``noise_normals`` enables paired common-random-number simulations.
    """
    config.validate()
    n = instance.n
    rng = np.random.default_rng(seed)
    x = (
        _initial_state(config, n, rng)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    if x.shape != (n,):
        raise ValueError(f"initial_state must have shape {(n,)}, got {x.shape}")
    if noise_normals is not None and noise_normals.shape != (config.steps, n):
        raise ValueError(f"noise_normals must have shape {(config.steps, n)}")

    record_steps = list(range(0, config.steps + 1, config.record_stride))
    if record_steps[-1] != config.steps:
        record_steps.append(config.steps)
    record_lookup = set(record_steps)
    n_records = len(record_steps)

    trajectory = np.empty((n_records, n), dtype=np.float64)
    spin_trajectory = np.empty((n_records, n), dtype=np.int8)
    scalar_names = (
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
    )
    values = {name: np.empty(n_records, dtype=np.float64) for name in scalar_names}
    steps_out = np.asarray(record_steps, dtype=np.int64)

    spins = decoded_spins(x)
    initial_spins = spins.copy()
    best_spins = spins.copy()
    best_energy = decoded_energy_density(instance.couplings, spins)
    cumulative_flips = 0
    recent_flips: deque[int] = deque(maxlen=config.flip_window_steps)
    record_index = 0

    def record(step: int, gain: float, temperature: float) -> None:
        nonlocal record_index, best_energy, best_spins
        current_energy = decoded_energy_density(instance.couplings, spins)
        if current_energy < best_energy:
            best_energy = current_energy
            best_spins = spins.copy()
        kde, bandwidth = p0_gaussian_kde(x, config.kde_bandwidth_factor)
        hist = p0_histogram(x, config.p0_bin_width)
        elapsed_window = len(recent_flips) * config.dt
        flip_rate = sum(recent_flips) / (n * elapsed_window) if elapsed_window else 0.0
        trajectory[record_index] = x
        spin_trajectory[record_index] = spins
        values["energy"][record_index] = current_energy
        values["best_energy"][record_index] = best_energy
        values["residual"][record_index] = current_energy - reference_energy
        values["best_residual"][record_index] = best_energy - reference_energy
        values["p0_kde"][record_index] = kde
        values["p0_hist"][record_index] = hist
        values["kde_bandwidth"][record_index] = bandwidth
        values["flip_rate"][record_index] = flip_rate
        values["sqrt_dt_mu"][record_index] = np.sqrt(config.dt) * flip_rate
        values["p0_mu_prediction"][record_index] = np.sqrt(temperature / np.pi) * kde
        values["cumulative_flips"][record_index] = cumulative_flips
        values["amplitude_mean"][record_index] = np.mean(x)
        values["amplitude_variance"][record_index] = np.var(x)
        values["overlap_initial"][record_index] = np.mean(spins * initial_spins)
        values["overlap_best"][record_index] = np.mean(spins * best_spins)
        values["soft_fraction"][record_index] = np.mean(np.abs(x) < 0.5 * config.p0_bin_width)
        values["gain"][record_index] = gain
        values["temperature"][record_index] = temperature
        record_index += 1

    gain, temperature = schedule(0, config.steps)
    record(0, gain, temperature)
    for integration_step in range(config.steps):
        gain, temperature = schedule(integration_step, config.steps)
        drift = _drift(x, spins, instance.couplings, gain, config)
        if external_fields and integration_step in external_fields:
            drift = drift + np.asarray(external_fields[integration_step])
        normal = rng.normal(size=n) if noise_normals is None else noise_normals[integration_step]
        x = x + config.dt * drift + np.sqrt(2.0 * temperature * config.dt) * normal
        if config.model_kind in {"digcim", "simcim"} and config.project_after_step:
            x = np.clip(x, -config.boundary, config.boundary)
        new_spins = decoded_spins(x, spins)
        flips = int(np.count_nonzero(new_spins != spins))
        recent_flips.append(flips)
        cumulative_flips += flips
        spins = new_spins
        completed_step = integration_step + 1
        if completed_step in record_lookup:
            record_gain, record_temperature = schedule(completed_step, config.steps)
            record(completed_step, record_gain, record_temperature)

    return SimulationResult(
        time=steps_out * config.dt,
        step=steps_out,
        x=trajectory,
        spins=spin_trajectory,
        **values,
        reference_energy=float(reference_energy),
        config=config,
        seed=seed,
        schedule_name=str(getattr(schedule, "name", type(schedule).__name__)),
    )


def estimate_response(
    instance: SKInstance,
    schedule: Schedule,
    config: SimulationConfig,
    seed: int,
    reference_energy: float,
    perturb_record_indices: list[int],
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the diagonal-mean G(t,t') with paired simulations and a probe vector."""
    rng = np.random.default_rng(seed)
    initial_state = _initial_state(config, instance.n, rng)
    noises = rng.normal(size=(config.steps, instance.n))
    probe = rng.choice(np.asarray([-1.0, 1.0]), size=instance.n)
    baseline = simulate(
        instance,
        schedule,
        config,
        seed,
        reference_energy,
        initial_state=initial_state,
        noise_normals=noises,
    )
    response = np.full((len(perturb_record_indices), len(baseline.time)), np.nan)
    perturb_times = np.empty(len(perturb_record_indices))
    for row, record_index in enumerate(perturb_record_indices):
        perturb_step = int(baseline.step[record_index])
        perturb_times[row] = perturb_step * config.dt
        perturbed = simulate(
            instance,
            schedule,
            config,
            seed,
            reference_energy,
            initial_state=initial_state,
            noise_normals=noises,
            external_fields={perturb_step: epsilon * probe},
        )
        delta = perturbed.x - baseline.x
        response[row] = (delta @ probe) / (instance.n * epsilon * config.dt)
        response[row, baseline.step < perturb_step] = np.nan
    return baseline.time, perturb_times, response
