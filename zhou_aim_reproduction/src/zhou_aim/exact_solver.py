from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observables import decoded_energy_density
from .sk_instance import SKInstance


@dataclass(frozen=True)
class ExactLandscape:
    spins: np.ndarray
    energies: np.ndarray
    ground_energy: float
    ground_spins: np.ndarray
    hamming_to_ground: np.ndarray


def enumerate_landscape(instance: SKInstance, max_n: int = 22) -> ExactLandscape:
    n = instance.n
    if n > max_n:
        raise ValueError(f"Exact enumeration limited to N <= {max_n}; received N={n}")
    state_ids = np.arange(1 << n, dtype=np.uint64)
    bits = ((state_ids[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
    spins = 2 * bits - 1
    energies = -0.5 * np.einsum("bi,ij,bj->b", spins, instance.couplings, spins) / n
    best_index = int(np.argmin(energies))
    ground_spins = spins[best_index].copy()
    hamming = np.count_nonzero(spins != ground_spins, axis=1)
    return ExactLandscape(
        spins=spins,
        energies=energies,
        ground_energy=float(energies[best_index]),
        ground_spins=ground_spins,
        hamming_to_ground=hamming,
    )


def brute_force_ground_energy(instance: SKInstance, max_n: int = 22) -> float:
    return enumerate_landscape(instance, max_n=max_n).ground_energy


def trajectory_state_ids(spins: np.ndarray) -> np.ndarray:
    n = spins.shape[1]
    if n > 63:
        raise ValueError("Integer state IDs are only supported for N <= 63")
    bits = (spins > 0).astype(np.uint64)
    return bits @ (np.uint64(1) << np.arange(n, dtype=np.uint64))


def verify_energy(instance: SKInstance, landscape: ExactLandscape) -> bool:
    sample = np.linspace(0, len(landscape.energies) - 1, min(17, len(landscape.energies))).astype(
        int
    )
    return all(
        np.isclose(
            decoded_energy_density(instance.couplings, landscape.spins[index]),
            landscape.energies[index],
        )
        for index in sample
    )
