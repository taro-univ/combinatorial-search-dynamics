from __future__ import annotations

import numpy as np

from zhou_aim.dynamics import SimulationConfig, SimulationResult, simulate
from zhou_aim.exact_solver import enumerate_landscape, verify_energy
from zhou_aim.observables import p0_gaussian_kde
from zhou_aim.schedules import LinearSchedule, PiecewiseLinearSchedule
from zhou_aim.sk_instance import SKInstance, generate_sk


def test_sk_reproducibility_and_roundtrip(tmp_path) -> None:
    first = generate_sk(32, seed=7)
    second = generate_sk(32, seed=7)
    assert np.array_equal(first.couplings, second.couplings)
    report = first.validation()
    assert report["symmetric"]
    assert report["diagonal_zero"]
    path = tmp_path / "instance.npz"
    first.save(path)
    loaded = SKInstance.load(path)
    assert np.array_equal(first.couplings, loaded.couplings)


def test_exact_landscape_energy() -> None:
    instance = generate_sk(8, seed=9)
    landscape = enumerate_landscape(instance)
    assert len(landscape.energies) == 2**8
    assert verify_energy(instance, landscape)
    assert np.count_nonzero(np.isclose(landscape.energies, landscape.ground_energy)) >= 2


def test_schedule_endpoints() -> None:
    linear = LinearSchedule(-2, 2, 1, 0.01)
    assert linear(0, 101) == (-2.0, 1.0)
    assert linear(100, 101) == (2.0, 0.01)
    path = PiecewiseLinearSchedule([(0, -2, 1), (0.5, 0, 0.2), (1, 2, 0.01)])
    assert np.allclose(path(50, 101), (0, 0.2))


def test_simulation_is_seed_reproducible_and_serializable(tmp_path) -> None:
    instance = generate_sk(10, seed=1)
    config = SimulationConfig(steps=40, record_stride=4)
    schedule = LinearSchedule(0, 0, 0.1, 0.1)
    first = simulate(instance, schedule, config, seed=2, reference_energy=-0.7)
    second = simulate(instance, schedule, config, seed=2, reference_energy=-0.7)
    assert np.array_equal(first.x, second.x)
    assert np.all(np.diff(first.best_energy) <= 1e-15)
    path = tmp_path / "result.npz"
    first.save(path)
    loaded = SimulationResult.load(path)
    assert np.array_equal(first.spins, loaded.spins)


def test_digcim_is_bounded() -> None:
    instance = generate_sk(10, seed=3)
    config = SimulationConfig(model_kind="digcim", steps=30, boundary=0.5)
    result = simulate(instance, LinearSchedule(-1, -1, 0.2, 0.2), config, 4, -0.7)
    assert np.max(np.abs(result.x)) <= 0.5


def test_kde_at_zero_for_normal_sample() -> None:
    sample = np.random.default_rng(123).normal(size=20000)
    estimate, _ = p0_gaussian_kde(sample)
    assert abs(estimate - 1 / np.sqrt(2 * np.pi)) < 0.03
