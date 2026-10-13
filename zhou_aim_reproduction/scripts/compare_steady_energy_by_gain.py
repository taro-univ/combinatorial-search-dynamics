from __future__ import annotations

import argparse
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from common import load_profile, print_summary, save_manifest, simulation_config, unique_run_dir

from zhou_aim.dynamics import simulate
from zhou_aim.observables import finite_size_ground_energy_mean
from zhou_aim.schedules import LinearSchedule
from zhou_aim.sk_instance import generate_sk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="medium", choices=["small", "medium", "full"])
    parser.add_argument("--run-id")
    parser.add_argument("--n", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--burn-steps", type=int, default=5000)
    parser.add_argument("--sample-steps", type=int, default=5000)
    parser.add_argument("--record-stride", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument(
        "--a-values",
        type=float,
        nargs="+",
        default=[-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument("--instance-limit", type=int, default=1)
    parser.add_argument("--run-limit", type=int, default=3)
    args = parser.parse_args()

    if args.burn_steps < 1 or args.sample_steps < 1:
        raise ValueError("burn-steps and sample-steps must be positive")

    profile = load_profile(args.profile)
    n = args.n or int(profile["model"]["n"])
    dt = args.dt or float(profile["integration"]["dt"])
    total_steps = args.burn_steps + args.sample_steps
    run_dir = unique_run_dir("steady_gain_comparison", args.run_id)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir()
    rows: list[dict[str, float | int | str]] = []
    resolved_configs = {}
    instance_seeds = profile["experiment"]["instance_seeds"][: args.instance_limit]
    run_seeds = profile["experiment"]["run_seeds"][: args.run_limit]
    reference_energy = finite_size_ground_energy_mean(n)

    for instance_seed in instance_seeds:
        instance = generate_sk(n, int(instance_seed), float(profile["model"]["j_scale"]))
        for run_seed in run_seeds:
            for gain in args.a_values:
                schedule = LinearSchedule(
                    gain, gain, args.temperature, args.temperature, name="fixed"
                )
                for model_kind in ("cim", "digcim"):
                    config = replace(
                        simulation_config(profile, model_kind=model_kind),
                        dt=dt,
                        steps=total_steps,
                        record_stride=args.record_stride,
                    )
                    resolved_configs[model_kind] = asdict(config)
                    result = simulate(
                        instance,
                        schedule,
                        config,
                        seed=int(run_seed),
                        reference_energy=reference_energy,
                    )
                    steady_mask = result.step >= args.burn_steps
                    rows.append(
                        {
                            "model": model_kind,
                            "gain": gain,
                            "temperature": args.temperature,
                            "steady_energy": float(np.mean(result.energy[steady_mask])),
                            "steady_energy_std_time": float(
                                np.std(result.energy[steady_mask], ddof=1)
                            ),
                            "final_energy": float(result.energy[-1]),
                            "instance_seed": int(instance_seed),
                            "run_seed": int(run_seed),
                            "n": n,
                            "dt": dt,
                            "physical_duration": total_steps * dt,
                            "tail_flips_per_spin": float(
                                result.cumulative_flips[-1]
                                - result.cumulative_flips[steady_mask][0]
                            )
                            / n,
                            "tail_endpoint_overlap": float(
                                np.mean(result.spins[steady_mask][0] * result.spins[-1])
                            ),
                        }
                    )

    runs = pd.DataFrame(rows)
    runs.to_csv(run_dir / "steady_energy_runs.csv", index=False)
    summary = runs.groupby(["model", "gain"], as_index=False).agg(
        mean=("steady_energy", "mean"),
        std=("steady_energy", "std"),
        count=("steady_energy", "count"),
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    summary.to_csv(run_dir / "steady_energy_summary.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"cim": "tab:red", "digcim": "black"}
    labels = {"cim": "CIM", "digcim": "digCIM"}
    for model_kind, frame in summary.groupby("model", sort=False):
        frame = frame.sort_values("gain")
        ax.plot(
            frame["gain"],
            frame["mean"],
            marker="o",
            color=colors[model_kind],
            label=labels[model_kind],
        )
        ax.fill_between(
            frame["gain"],
            frame["mean"] - frame["ci95"],
            frame["mean"] + frame["ci95"],
            color=colors[model_kind],
            alpha=0.18,
        )
    ax.axhline(reference_energy, color="0.4", linestyle="--", label="finite-size mean reference")
    ax.set(xlabel="gain a", ylabel="finite-time tail decoded energy density")
    ax.set_title(
        f"CIM vs digCIM: fixed T={args.temperature:g}, N={n}\n"
        f"dt={dt:g}, physical time={total_steps * dt:g}"
    )
    ax.legend()
    fig.savefig(figure_dir / "steady_energy_by_gain.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    details = {
        "n": n,
        "dt": dt,
        "burn_steps": args.burn_steps,
        "sample_steps": args.sample_steps,
        "record_stride": args.record_stride,
        "temperature": args.temperature,
        "a_values": args.a_values,
        "instance_seeds": instance_seeds,
        "run_seeds": run_seeds,
        "run_count": len(runs),
        "resolved_simulation_configs": resolved_configs,
        "physical_duration": total_steps * dt,
        "temperature_protocol": "fixed low temperature from initialization; no annealing",
        "steady_energy_definition": "mean decoded energy over records after burn_steps",
        "equilibrium_claim": False,
        "uncertainty": "approximate normal CI over runs; conditional on sampled instances",
    }
    save_manifest(run_dir, profile, details)
    print_summary({"run_dir": str(run_dir), **details})


if __name__ == "__main__":
    main()
