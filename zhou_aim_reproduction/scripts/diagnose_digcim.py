from __future__ import annotations

import argparse
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_profile, print_summary, save_manifest, simulation_config, unique_run_dir
from matplotlib.ticker import NullLocator

from zhou_aim.dynamics import SimulationConfig, simulate
from zhou_aim.exact_solver import brute_force_ground_energy
from zhou_aim.observables import finite_size_ground_energy_mean
from zhou_aim.provenance import write_json
from zhou_aim.schedules import LinearSchedule, PiecewiseLinearSchedule
from zhou_aim.sk_instance import generate_sk


def save_exact_controls(run_dir, run_seeds):
    """Check whether a frozen, single-flip-stable result is actually a ground state."""
    instance = generate_sk(16, 11)
    config = SimulationConfig(model_kind="digcim", steps=10000, record_stride=100)
    ground = brute_force_ground_energy(instance)
    rows = []
    for seed in run_seeds:
        result = simulate(instance, LinearSchedule(-1, -1, 1e-5, 1e-5), config, seed, ground)
        spins = result.spins[-1]
        edge_energy = (
            -sum(
                instance.couplings[i, j] * spins[i] * spins[j]
                for i in range(instance.n)
                for j in range(i + 1, instance.n)
            )
            / instance.n
        )
        if not np.isclose(edge_energy, result.energy[-1]):
            raise AssertionError("Matrix and independent edge-sum energies disagree")
        rows.append(
            {
                "n": instance.n,
                "instance_seed": instance.seed,
                "run_seed": seed,
                "exact_ground_energy": ground,
                "final_energy": result.energy[-1],
                "edge_sum_energy": edge_energy,
                "unstable_spins": np.count_nonzero(spins * (instance.couplings @ spins) < 0),
                "last_38_time_units_flips": (
                    result.cumulative_flips[-1] - result.cumulative_flips[-20]
                ),
            }
        )
        result.save(run_dir / "trajectories" / f"exact_control_{seed}.npz")
    pd.DataFrame(rows).to_csv(run_dir / "small_exact_checks.csv", index=False)
    write_json(
        run_dir / "small_exact_config.json",
        {
            "config": asdict(config),
            "gain": -1,
            "temperature": 1e-5,
            "n": instance.n,
            "instance_seed": instance.seed,
            "run_seeds": run_seeds,
        },
    )


def plot_summary(summary, run_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    settings = [
        ("gain_protocol", "gain", "Gain / temperature protocol", "Gain a"),
        ("dt", "dt", "Time step (fixed physical duration)", "Time step dt"),
        ("size", "n", "Size (gap to finite-size mean reference)", "Number of spins N"),
        ("runtime", "duration", "Runtime", "Physical duration"),
    ]
    names = {"fixed_low": "Fixed low T", "linear_cool_hold": "Cooling + low T hold"}
    for ax, (sweep, xkey, title, xlabel) in zip(axes.flat, settings, strict=True):
        frame = summary[summary.sweep == sweep]
        ykey = "final_reference_gap_mean" if sweep == "size" else "final_energy_mean"
        group_keys = ["protocol"] if sweep == "gain_protocol" else ["protocol", "gain"]
        for key, group in frame.groupby(group_keys):
            group = group.sort_values(xkey)
            label = names[key[0]]
            if sweep == "dt":
                label += f", a={key[1]:g}"
            ax.errorbar(
                group[xkey],
                group[ykey],
                yerr=group.final_energy_std.fillna(0),
                fmt="o-",
                capsize=3,
                label=label,
            )
        if sweep in {"dt", "size", "runtime"}:
            ax.set_xscale("log")
            ticks = sorted(frame[xkey].unique())
            ax.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel="Final decoded energy - reference"
            if sweep == "size"
            else "Final decoded energy",
        )
        ax.legend(fontsize=8)
    fig.suptitle("digCIM, B=1: mean +/- run SD; one SK instance per N", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "diagnosis.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate digCIM size, dt, runtime and cooling effects"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--base-n", type=int, default=512)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 2000])
    parser.add_argument("--long-n", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--long-duration", type=float, default=2000.0)
    parser.add_argument("--dt-values", type=float, nargs="+", default=[0.1, 0.02, 0.005])
    parser.add_argument("--gains", type=float, nargs="+", default=[-3.0, -1.0, 0.0, 0.5, 1.0])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()
    if min([args.base_n, args.long_n, *args.sizes]) < 2:
        raise ValueError("All sizes must be at least 2")
    if args.duration <= 0 or args.long_duration < args.duration:
        raise ValueError("Require 0 < duration <= long-duration")
    if min(args.dt_values) <= 0:
        raise ValueError("dt-values must be positive")
    for duration in (args.duration, args.long_duration):
        for dt in (0.02, *args.dt_values):
            if not np.isclose(duration / dt, round(duration / dt)):
                raise ValueError("Durations must contain an integer number of steps for every dt")
    fine_dt = min(args.dt_values)
    for dt in args.dt_values:
        if not np.isclose(dt / fine_dt, round(dt / fine_dt)):
            raise ValueError("All dt-values must be integer multiples of the smallest dt")

    profile = load_profile("full")
    base = simulation_config(profile, model_kind="digcim")
    run_dir = unique_run_dir("digcim_diagnosis", args.run_id)
    (run_dir / "trajectories").mkdir()
    (run_dir / "figures").mkdir()
    details = {
        **vars(args),
        "temperature_low": 1e-5,
        "temperature_high": 1.0,
        "cooling_fraction": 0.8,
        "boundary": base.boundary,
        "xi": base.xi,
        "j_scale": profile["model"]["j_scale"],
        "tail_fraction": 0.2,
        "dt_comparison": "fixed physical duration; shared initial state and Brownian increments",
        "other_comparisons": "same instance and run seeds; common noise for equal dt",
        "reference": "finite-size disorder-mean approximation, not an exact instance optimum",
    }
    save_manifest(run_dir, profile, details)
    save_exact_controls(run_dir, args.run_seeds)
    rows: list[dict[str, object]] = []

    def run(instance, seed, sweep, gain, protocol, duration, dt=0.02, **kwargs):
        steps = round(duration / dt)
        config = replace(
            base,
            dt=dt,
            steps=steps,
            record_stride=max(1, round(duration / (100 * dt))),
            flip_window_steps=max(1, round(2 / dt)),
        )
        if protocol == "fixed_low":
            schedule = LinearSchedule(gain, gain, 1e-5, 1e-5, name=protocol)
        else:
            schedule = PiecewiseLinearSchedule(
                [(0, gain, 1.0), (0.8, gain, 1e-5), (1, gain, 1e-5)], name=protocol
            )
        result = simulate(
            instance, schedule, config, seed, finite_size_ground_energy_mean(instance.n), **kwargs
        )
        tail = result.time >= 0.8 * duration - 1e-10
        final_spins = result.spins[-1]
        stability = final_spins * (instance.couplings @ final_spins)
        row = {
            "sweep": sweep,
            "n": instance.n,
            "instance_seed": instance.seed,
            "run_seed": seed,
            "gain": gain,
            "protocol": protocol,
            "dt": dt,
            "steps": steps,
            "duration": duration,
            "final_energy": result.energy[-1],
            "tail_energy": np.mean(result.energy[tail]),
            "tail_energy_std": np.std(result.energy[tail]),
            "best_recorded_energy": result.best_energy[-1],
            "reference_energy": result.reference_energy,
            "final_reference_gap": result.energy[-1] - result.reference_energy,
            "tail_flips_per_spin": (result.cumulative_flips[-1] - result.cumulative_flips[tail][0])
            / instance.n,
            "tail_endpoint_overlap": np.mean(result.spins[tail][0] * final_spins),
            "final_unstable_spin_fraction": np.mean(stability < 0),
            "final_boundary_fraction": np.mean(np.abs(result.x[-1]) >= base.boundary),
        }
        rows.append(row)
        result.save(run_dir / "trajectories" / f"run_{len(rows):04d}.npz")
        pd.DataFrame(rows).to_csv(run_dir / "runs.csv", index=False)
        print(
            f"{len(rows):03d} {sweep} N={instance.n} a={gain:g} {protocol} "
            f"dt={dt:g} t={duration:g} seed={seed} E={row['final_energy']:.6f} "
            f"tail_flips/spin={row['tail_flips_per_spin']:.3f}",
            flush=True,
        )

    for instance_seed in args.instance_seeds:
        instance = generate_sk(args.base_n, instance_seed)
        instance.save(run_dir / f"instance_n{instance.n}_seed{instance_seed}.npz")
        for seed in args.run_seeds:
            # Aggregate fine increments: sqrt(dt) * eta has exactly the same sum at every dt.
            rng = np.random.default_rng(seed)
            initial = rng.normal(0, base.initial_std, size=instance.n)
            fine_noise = rng.normal(size=(round(args.duration / fine_dt), instance.n))
            for dt in args.dt_values:
                ratio = round(dt / fine_dt)
                noise = fine_noise.reshape(-1, ratio, instance.n).sum(axis=1) / np.sqrt(ratio)
                for gain in (-1.0, 0.5):
                    run(
                        instance,
                        seed,
                        "dt",
                        gain,
                        "fixed_low",
                        args.duration,
                        dt,
                        initial_state=initial,
                        noise_normals=noise,
                    )
            del fine_noise, noise
            for gain in args.gains:
                for protocol in ("fixed_low", "linear_cool_hold"):
                    run(instance, seed, "gain_protocol", gain, protocol, args.duration)
            for duration in (args.duration, args.long_duration):
                for protocol in ("fixed_low", "linear_cool_hold"):
                    run(instance, seed, "runtime", -1.0, protocol, duration)

        for n in args.sizes:
            size_instance = instance if n == instance.n else generate_sk(n, instance_seed)
            size_instance.save(run_dir / f"instance_n{n}_seed{instance_seed}.npz")
            for seed in args.run_seeds:
                for protocol in ("fixed_low", "linear_cool_hold"):
                    run(size_instance, seed, "size", -1.0, protocol, args.duration)

        long_instance = generate_sk(args.long_n, instance_seed)
        long_instance.save(run_dir / f"instance_n{long_instance.n}_seed{instance_seed}.npz")
        for seed in args.run_seeds:
            run(long_instance, seed, "long_anneal", -1.0, "linear_cool_hold", args.long_duration)

    runs = pd.DataFrame(rows)
    keys = ["sweep", "n", "gain", "protocol", "dt", "duration"]
    summary = runs.groupby(keys, as_index=False).agg(
        final_energy_mean=("final_energy", "mean"),
        final_energy_std=("final_energy", "std"),
        tail_energy_mean=("tail_energy", "mean"),
        tail_flips_per_spin_mean=("tail_flips_per_spin", "mean"),
        final_reference_gap_mean=("final_reference_gap", "mean"),
        unstable_fraction_mean=("final_unstable_spin_fraction", "mean"),
        count=("final_energy", "count"),
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    plot_summary(summary, run_dir)
    save_manifest(run_dir, profile, {**details, "run_count": len(rows), "complete": True})
    print_summary({"run_dir": str(run_dir), "run_count": len(rows)})


if __name__ == "__main__":
    main()
