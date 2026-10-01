from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from common import (
    load_profile,
    print_summary,
    result_frame,
    save_manifest,
    simulation_config,
    unique_run_dir,
)

from zhou_aim.dynamics import SimulationResult, estimate_response, simulate
from zhou_aim.exact_solver import enumerate_landscape, verify_energy
from zhou_aim.observables import correlation_matrix
from zhou_aim.schedules import schedule_from_dict
from zhou_aim.sk_instance import SKInstance, generate_sk
from zhou_aim.visualization import (
    animate_diagnostics,
    plot_diagnostics,
    plot_distribution_evolution,
    plot_pca_trajectory,
    plot_response,
    plot_small_landscape,
    plot_state_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--run-id")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    run_dir = unique_run_dir("smoke", args.run_id)
    raw = run_dir / "raw"
    figures = run_dir / "figures"
    raw.mkdir()
    figures.mkdir()

    instance = generate_sk(n=12, seed=11, j_scale=1.0)
    instance.save(raw / "sk_n12_seed11.npz")
    reloaded = SKInstance.load(raw / "sk_n12_seed11.npz")
    regenerated = generate_sk(n=12, seed=11, j_scale=1.0)
    assert np.array_equal(instance.couplings, reloaded.couplings)
    assert np.array_equal(instance.couplings, regenerated.couplings)
    validation = instance.validation()
    assert validation["symmetric"] and validation["diagonal_zero"]

    landscape = enumerate_landscape(instance)
    assert verify_energy(instance, landscape)
    config = replace(
        simulation_config(profile),
        steps=800,
        record_stride=4,
        flip_window_steps=20,
    )
    schedule = schedule_from_dict(profile["temperature_schedule"])
    result = simulate(
        instance, schedule, config, seed=101, reference_energy=landscape.ground_energy
    )
    repeated = simulate(
        instance, schedule, config, seed=101, reference_energy=landscape.ground_energy
    )
    assert np.array_equal(result.x, repeated.x)
    assert np.array_equal(result.energy, repeated.energy)
    result.save(raw / "temperature_result.npz")
    roundtrip = SimulationResult.load(raw / "temperature_result.npz")
    assert np.array_equal(result.x, roundtrip.x)
    result_frame(result, schedule="temperature", run_id="smoke").to_csv(
        run_dir / "trajectory.csv", index=False
    )
    np.savez_compressed(
        raw / "two_time_correlation.npz", q=correlation_matrix(result.x), time=result.time
    )

    response_config = replace(config, steps=240, record_stride=4)
    response_schedule = schedule_from_dict(profile["fixed_schedule"])
    response_time, perturb_time, response = estimate_response(
        instance,
        response_schedule,
        response_config,
        seed=919,
        reference_energy=landscape.ground_energy,
        perturb_record_indices=[5, 20, 40],
        epsilon=1e-4,
    )
    np.savez_compressed(
        raw / "paired_response.npz",
        time=response_time,
        perturb_time=perturb_time,
        response=response,
        epsilon=1e-4,
    )

    plot_diagnostics(result, figures / "diagnostics.png")
    plot_state_heatmaps(result, figures / "state_heatmaps.png")
    plot_distribution_evolution(result, figures / "distribution_evolution.png")
    plot_small_landscape(landscape, result, figures / "exact_landscape.png")
    plot_pca_trajectory(result, figures / "pca_trajectory.png")
    plot_response(response_time, perturb_time, response, figures / "paired_response.png")
    animate_diagnostics(result, figures / "representative_run.gif", max_frames=80)

    details = {
        "instance_validation": validation,
        "exact_ground_energy": landscape.ground_energy,
        "final_energy": float(result.energy[-1]),
        "best_energy": float(result.best_energy[-1]),
        "visited_states": len(np.unique(result.spins, axis=0)),
        "checks": {
            "seed_reproducible": True,
            "instance_roundtrip": True,
            "result_roundtrip": True,
            "exact_energy_consistent": True,
            "paired_response_finite_after_perturbation": bool(np.isfinite(response[:, -1]).all()),
        },
    }
    save_manifest(run_dir, profile, details)
    print_summary({"run_dir": str(run_dir), **details})


if __name__ == "__main__":
    main()
