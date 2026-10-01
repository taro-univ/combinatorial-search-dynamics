from __future__ import annotations

import numpy as np


def decoded_spins(x: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    spins = np.sign(x).astype(np.int8)
    zeros = spins == 0
    if previous is None:
        spins[zeros] = 1
    else:
        spins[zeros] = previous[zeros]
    return spins


def decoded_energy_density(couplings: np.ndarray, spins: np.ndarray) -> float:
    return float(-0.5 * spins @ couplings @ spins / len(spins))


def p0_gaussian_kde(x: np.ndarray, bandwidth_factor: float = 1.0) -> tuple[float, float]:
    """Return Gaussian KDE at zero and the effective bandwidth.

    Scott's rule is used, with a numerical floor for nearly degenerate populations.
    """
    n = len(x)
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    scale = max(std, float(np.max(np.abs(x))) * 1e-6, 1e-8)
    bandwidth = max(scale * n ** (-1.0 / 5.0) * bandwidth_factor, 1e-10)
    z = x / bandwidth
    density = np.exp(-0.5 * z * z).mean() / (np.sqrt(2.0 * np.pi) * bandwidth)
    return float(density), float(bandwidth)


def p0_histogram(x: np.ndarray, bin_width: float) -> float:
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    return float(np.count_nonzero(np.abs(x) < 0.5 * bin_width) / (len(x) * bin_width))


def correlation_matrix(trajectory: np.ndarray) -> np.ndarray:
    return trajectory @ trajectory.T / trajectory.shape[1]


def thermodynamic_ground_energy() -> float:
    return -0.7632


def finite_size_ground_energy_mean(n: int) -> float:
    return thermodynamic_ground_energy() + 0.701 * n ** (-2.0 / 3.0)
