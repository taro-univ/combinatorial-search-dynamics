from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_profile, print_summary, save_manifest, simulation_config, unique_run_dir

from zhou_aim.dynamics import simulate
from zhou_aim.exact_solver import brute_force_ground_energy
from zhou_aim.observables import finite_size_ground_energy_mean, p0_gaussian_kde, p0_histogram
from zhou_aim.schedules import schedule_from_dict
from zhou_aim.sk_instance import generate_sk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    run_dir = unique_run_dir("validation", args.run_id)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir()
    schedule = schedule_from_dict(profile["temperature_schedule"])
    base = replace(
        simulation_config(profile), steps=min(int(profile["integration"]["steps"]), 1200)
    )

    rows: list[dict[str, object]] = []
    coupling_rows = []
    sizes = [int(value) for value in profile["experiment"]["sizes"]]
    if args.profile == "small":
        sizes = [12, 32, 64]
    seeds = [int(value) for value in profile["experiment"]["instance_seeds"][:3]]
    representative = None
    for n in sizes:
        for seed in seeds:
            instance = generate_sk(n, seed, float(profile["model"]["j_scale"]))
            coupling_rows.append(instance.validation())
            reference = (
                brute_force_ground_energy(instance)
                if n <= 20
                else finite_size_ground_energy_mean(n)
            )
            result = simulate(instance, schedule, base, seed + 1000, reference)
            representative = representative or result
            rows.append(
                {
                    "validation": "size_and_seed",
                    "n": n,
                    "instance_seed": seed,
                    "run_seed": seed + 1000,
                    "setting": "default",
                    "final_energy": result.energy[-1],
                    "best_energy": result.best_energy[-1],
                    "reference_energy": reference,
                    "final_residual": result.best_residual[-1],
                }
            )

    n_dt = max(32, min(64, sizes[-1]))
    instance_dt = generate_sk(n_dt, 4242)
    physical_time = 12.0
    for dt in (0.04, 0.02, 0.01):
        for run_seed in (81, 82, 83):
            config = replace(
                base,
                dt=dt,
                steps=round(physical_time / dt),
                record_stride=max(1, round(0.2 / dt)),
                flip_window_steps=max(1, round(0.4 / dt)),
            )
            result = simulate(
                instance_dt,
                schedule,
                config,
                run_seed,
                finite_size_ground_energy_mean(n_dt),
            )
            rows.append(
                {
                    "validation": "dt_convergence",
                    "n": n_dt,
                    "instance_seed": 4242,
                    "run_seed": run_seed,
                    "setting": dt,
                    "final_energy": result.energy[-1],
                    "best_energy": result.best_energy[-1],
                    "reference_energy": result.reference_energy,
                    "final_residual": result.best_residual[-1],
                }
            )

    instance_init = generate_sk(32, 7171)
    for initialization in ("gaussian", "uniform", "zeros"):
        config = replace(base, initialization=initialization, steps=min(base.steps, 800))
        result = simulate(
            instance_init,
            schedule,
            config,
            313,
            finite_size_ground_energy_mean(instance_init.n),
        )
        rows.append(
            {
                "validation": "initialization",
                "n": 32,
                "instance_seed": 7171,
                "run_seed": 313,
                "setting": initialization,
                "final_energy": result.energy[-1],
                "best_energy": result.best_energy[-1],
                "reference_energy": result.reference_energy,
                "final_residual": result.best_residual[-1],
            }
        )

    instance_window = generate_sk(32, 8181)
    for window in (5, 20, 80):
        config = replace(base, steps=min(base.steps, 800), flip_window_steps=window)
        result = simulate(
            instance_window,
            schedule,
            config,
            414,
            finite_size_ground_energy_mean(instance_window.n),
        )
        rows.append(
            {
                "validation": "flip_window",
                "n": 32,
                "instance_seed": 8181,
                "run_seed": 414,
                "setting": window,
                "final_energy": result.energy[-1],
                "best_energy": result.best_energy[-1],
                "reference_energy": result.reference_energy,
                "final_residual": result.best_residual[-1],
                "final_flip_rate": result.flip_rate[-1],
            }
        )

    assert representative is not None
    p0_rows = []
    for time_index in np.linspace(0, len(representative.time) - 1, 7).astype(int):
        x = representative.x[time_index]
        for bandwidth_factor in (0.5, 1.0, 2.0):
            density, bandwidth = p0_gaussian_kde(x, bandwidth_factor)
            p0_rows.append(
                {
                    "time": representative.time[time_index],
                    "method": "kde",
                    "parameter": bandwidth_factor,
                    "p0": density,
                    "effective_bandwidth": bandwidth,
                }
            )
        for bin_width in (0.05, 0.1, 0.2):
            p0_rows.append(
                {
                    "time": representative.time[time_index],
                    "method": "histogram",
                    "parameter": bin_width,
                    "p0": p0_histogram(x, bin_width),
                    "effective_bandwidth": np.nan,
                }
            )

    validation = pd.DataFrame(rows)
    coupling = pd.DataFrame(coupling_rows)
    p0 = pd.DataFrame(p0_rows)
    validation.to_csv(run_dir / "numerical_validation.csv", index=False)
    coupling.to_csv(run_dir / "coupling_validation.csv", index=False)
    p0.to_csv(run_dir / "p0_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    size_frame = validation[validation["validation"] == "size_and_seed"]
    for n, group in size_frame.groupby("n"):
        axes[0].scatter([n] * len(group), group["best_energy"], alpha=0.8)
    axes[0].set(xlabel="N", ylabel="best energy density", title="size / disorder dependence")
    dt_frame = validation[validation["validation"] == "dt_convergence"]
    dt_stats = dt_frame.groupby("setting")["best_energy"].agg(["mean", "std"]).reset_index()
    axes[1].errorbar(dt_stats["setting"], dt_stats["mean"], yerr=dt_stats["std"], marker="o")
    axes[1].set(xlabel="dt", ylabel="best energy density", title="time-step convergence")
    for (method, parameter), group in p0.groupby(["method", "parameter"]):
        axes[2].plot(group["time"], group["p0"], marker="o", label=f"{method}:{parameter}")
    axes[2].set(xlabel="time", ylabel="P(0)", title="estimator sensitivity")
    axes[2].legend(fontsize=6)
    fig.savefig(figure_dir / "validation_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    dt_means = dt_frame.groupby("setting")["best_energy"].mean().to_dict()
    details = {
        "couplings_all_symmetric": bool(coupling["symmetric"].all()),
        "couplings_all_diagonal_zero": bool(coupling["diagonal_zero"].all()),
        "variance_ratio_range": [
            float(coupling["variance_ratio"].min()),
            float(coupling["variance_ratio"].max()),
        ],
        "dt_mean_best_energy": {str(key): float(value) for key, value in dt_means.items()},
        "validation_rows": len(validation),
        "p0_sensitivity_rows": len(p0),
    }
    save_manifest(run_dir, profile, details)
    print_summary({"run_dir": str(run_dir), **details})


if __name__ == "__main__":
    main()
