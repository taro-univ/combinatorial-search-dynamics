from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .dynamics import SimulationResult
from .exact_solver import ExactLandscape, trajectory_state_ids


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(result: SimulationResult, path: str | Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
    time = result.time
    axes[0, 0].plot(time, result.energy, label="decoded")
    axes[0, 0].plot(time, result.best_energy, label="best-so-far")
    axes[0, 0].axhline(result.reference_energy, color="black", ls="--", lw=1, label="reference")
    axes[0, 0].set_ylabel("energy density")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(time, result.gain, label="a")
    axes[0, 1].plot(time, result.temperature, label="T")
    axes[0, 1].set_ylabel("control")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].semilogy(time[1:], np.maximum(result.flip_rate[1:], 1e-12), label="flip rate μ")
    axes[1, 0].semilogy(time[1:], np.maximum(result.p0_kde[1:], 1e-12), label="KDE P(0)")
    axes[1, 0].set_ylabel("activity / density")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(time[1:], result.sqrt_dt_mu[1:], label=r"$\sqrt{dt}\,\mu$")
    axes[1, 1].plot(time[1:], result.p0_mu_prediction[1:], label=r"$\sqrt{T/\pi}P(0)$")
    axes[1, 1].set_ylabel("Eq. (6) terms")
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].plot(time, result.amplitude_mean, label="mean x")
    axes[2, 0].plot(time, result.amplitude_variance, label="var x")
    axes[2, 0].plot(time, result.soft_fraction, label="near-zero fraction")
    axes[2, 0].set_xlabel("time")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(time, result.overlap_initial, label="spin overlap with initial")
    axes[2, 1].plot(time, result.overlap_best, label="spin overlap with current best")
    axes[2, 1].set_xlabel("time")
    axes[2, 1].legend(fontsize=8)
    fig.suptitle(f"{result.config.model_kind}: {result.schedule_name}")
    _save(fig, path)


def plot_state_heatmaps(result: SimulationResult, path: str | Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    extent = [result.time[0], result.time[-1], result.x.shape[1], 0]
    vmax = max(float(np.quantile(np.abs(result.x), 0.99)), 1e-8)
    im0 = axes[0].imshow(
        result.x.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )
    axes[0].set_ylabel("spin index")
    axes[0].set_title("soft-spin amplitude")
    fig.colorbar(im0, ax=axes[0], label="x")
    im1 = axes[1].imshow(
        result.spins.T,
        aspect="auto",
        interpolation="nearest",
        cmap="bwr",
        vmin=-1,
        vmax=1,
        extent=extent,
    )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("spin index")
    axes[1].set_title("decoded sign")
    fig.colorbar(im1, ax=axes[1], label="σ")
    _save(fig, path)


def plot_distribution_evolution(
    result: SimulationResult,
    path: str | Path,
    bins: int = 80,
    *,
    bound: float | None = None,
    density_vmax: float | None = None,
    relative_density: bool = False,
) -> None:
    bound = bound or max(float(np.quantile(np.abs(result.x), 0.995)), 0.25)
    edges = np.linspace(-bound, bound, bins + 1)
    hist = np.asarray([np.histogram(row, bins=edges, density=True)[0] for row in result.x])
    if relative_density:
        if density_vmax is None or density_vmax <= 0:
            raise ValueError("density_vmax must be positive for relative density")
        hist = hist / density_vmax
        colorbar_label = "relative density"
        image_vmax = 1.0
    else:
        colorbar_label = "density"
        image_vmax = density_vmax
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        extent=[result.time[0], result.time[-1], edges[0], edges[-1]],
        cmap="magma",
        vmin=0,
        vmax=image_vmax,
    )
    ax.axhline(0, color="cyan", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("soft-spin amplitude x")
    ax.set_title(r"Empirical $P_t(x)$ evolution")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    _save(fig, path)


def plot_response(
    time: np.ndarray, perturb_time: np.ndarray, response: np.ndarray, path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, time_prime in enumerate(perturb_time):
        ax.plot(time, response[index], label=f"t'={time_prime:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"paired estimate $G(t,t')$")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_small_landscape(
    landscape: ExactLandscape, result: SimulationResult, path: str | Path
) -> None:
    state_ids = trajectory_state_ids(result.spins)
    visited = np.zeros(len(landscape.energies), dtype=bool)
    visited[state_ids.astype(int)] = True
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(landscape.energies, bins=60, color="0.45")
    axes[0].axvline(landscape.ground_energy, color="red", ls="--")
    axes[0].set(xlabel="energy density", ylabel="state count", title="all Ising states")
    axes[1].scatter(
        landscape.hamming_to_ground[~visited],
        landscape.energies[~visited],
        s=4,
        alpha=0.2,
        label="unvisited",
    )
    axes[1].scatter(
        landscape.hamming_to_ground[visited],
        landscape.energies[visited],
        s=14,
        color="tab:orange",
        label="visited",
    )
    axes[1].set(xlabel="Hamming distance to one ground state", ylabel="energy density")
    axes[1].legend(fontsize=8)
    ground = landscape.ground_spins
    trajectory_hamming = np.count_nonzero(result.spins != ground, axis=1)
    axes[2].plot(result.time, trajectory_hamming, label="Hamming distance")
    update = np.r_[True, np.diff(result.best_energy) < 0]
    axes[2].scatter(
        result.time[update], trajectory_hamming[update], color="red", s=18, label="best update"
    )
    axes[2].set(xlabel="time", ylabel="distance", title="decoded trajectory")
    axes[2].legend(fontsize=8)
    _save(fig, path)


def plot_pca_trajectory(result: SimulationResult, path: str | Path) -> None:
    centered = result.x - result.x.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :2] * singular[:2]
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=result.time, cmap="viridis", s=16)
    ax.plot(projected[:, 0], projected[:, 1], color="0.5", lw=0.6, alpha=0.6)
    ax.set(xlabel="PC1", ylabel="PC2", title="Projection of high-dimensional soft-spin trajectory")
    fig.colorbar(scatter, ax=ax, label="time")
    _save(fig, path)


def plot_annealing_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grouped = summary.groupby(["schedule", "time"], sort=False)
    stats = grouped["energy"].agg(["mean", "std", "count"]).reset_index()
    stats["sem95"] = 1.96 * stats["std"].fillna(0) / np.sqrt(stats["count"])
    for name, frame in stats.groupby("schedule", sort=False):
        axes[0].plot(frame["time"], frame["mean"], label=name)
        axes[0].fill_between(
            frame["time"], frame["mean"] - frame["sem95"], frame["mean"] + frame["sem95"], alpha=0.2
        )
    final = summary.loc[summary.groupby(["schedule", "run_id"])["time"].idxmax()]
    schedules = list(final["schedule"].unique())
    axes[1].boxplot(
        [final.loc[final["schedule"] == name, "energy"] for name in schedules],
        tick_labels=schedules,
    )
    axes[0].set(xlabel="time", ylabel="decoded energy density", title="mean ± approximate 95% CI")
    axes[0].legend()
    axes[1].set(ylabel="final energy density", title="matched integration budget")
    _save(fig, path)


def plot_phase_map(
    frame: pd.DataFrame,
    paths: dict[str, np.ndarray],
    path: str | Path,
    freeze_points: dict[str, tuple[float, float]] | None = None,
) -> None:
    gains = np.sort(frame["gain"].unique())
    temperatures = np.sort(frame["temperature"].unique())
    shape = (len(temperatures), len(gains))
    energy = (
        frame.pivot(index="temperature", columns="gain", values="energy_mean")
        .loc[temperatures, gains]
        .to_numpy()
    )
    activity = (
        frame.pivot(index="temperature", columns="gain", values="flip_rate_mean")
        .loc[temperatures, gains]
        .to_numpy()
    )
    p0 = (
        frame.pivot(index="temperature", columns="gain", values="p0_mean")
        .loc[temperatures, gains]
        .to_numpy()
    )
    if energy.shape != shape:
        raise ValueError("Incomplete phase grid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    fields = [(energy, "decoded energy"), (p0, "KDE P(0)"), (activity, "flip rate μ")]
    for ax, (field, title) in zip(axes, fields, strict=True):
        contour = ax.contourf(gains, temperatures, field, levels=16, cmap="viridis")
        for name, points in paths.items():
            ax.plot(points[:, 0], points[:, 1], marker="o", ms=2.5, label=name)
            if freeze_points and name in freeze_points:
                point = freeze_points[name]
                ax.scatter(*point, marker="x", s=70, linewidths=2, label=f"{name} freeze")
        ax.set(xlabel="gain a", title=title)
        fig.colorbar(contour, ax=ax)
    axes[0].set_ylabel("temperature T")
    axes[-1].legend(fontsize=8)
    _save(fig, path)


def animate_diagnostics(result: SimulationResult, path: str | Path, max_frames: int = 160) -> None:
    indices = np.unique(
        np.linspace(0, len(result.time) - 1, min(max_frames, len(result.time))).astype(int)
    )
    bound = max(float(np.quantile(np.abs(result.x), 0.995)), 0.25)
    bins = np.linspace(-bound, bound, 50)
    fig, (ax_hist, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))
    ax_hist.set(xlim=(-bound, bound), xlabel="x", ylabel="density", title="soft-spin distribution")
    ax_energy.set(
        xlim=(result.time[0], result.time[-1]),
        ylim=(min(result.energy.min(), result.reference_energy) - 0.03, result.energy.max() + 0.03),
        xlabel="time",
        ylabel="energy density",
    )
    (energy_line,) = ax_energy.plot([], [], color="tab:blue")
    ax_energy.axhline(result.reference_energy, color="black", ls="--", lw=1)

    def update(frame: int):
        index = indices[frame]
        ax_hist.cla()
        ax_hist.hist(result.x[index], bins=bins, density=True, color="tab:purple", alpha=0.8)
        ax_hist.axvline(0, color="black", lw=0.8)
        ax_hist.set(
            xlim=(-bound, bound), xlabel="x", ylabel="density", title=f"t={result.time[index]:.2f}"
        )
        energy_line.set_data(result.time[: index + 1], result.energy[: index + 1])
        return (energy_line,)

    animation = FuncAnimation(fig, update, frames=len(indices), interval=70, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
