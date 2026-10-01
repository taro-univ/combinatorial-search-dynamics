from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from common import (
    load_profile,
    print_summary,
    result_frame,
    save_manifest,
    simulation_config,
    unique_run_dir,
)

from zhou_aim.dynamics import simulate
from zhou_aim.exact_solver import brute_force_ground_energy
from zhou_aim.observables import finite_size_ground_energy_mean
from zhou_aim.schedules import schedule_from_dict
from zhou_aim.sk_instance import generate_sk
from zhou_aim.visualization import (
    plot_annealing_comparison,
    plot_diagnostics,
    plot_distribution_evolution,
    plot_state_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--run-id")
    parser.add_argument("--n", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--record-stride", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--run-limit", type=int)
    parser.add_argument("--model-kind", choices=["cim", "digcim", "both"], default="both")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    n = args.n or int(profile["model"]["n"])
    model_kinds = ["cim", "digcim"] if args.model_kind == "both" else [args.model_kind]
    run_dir = unique_run_dir("dynamics", args.run_id)
    raw_dir = run_dir / "raw"
    figure_dir = run_dir / "figures"
    raw_dir.mkdir()
    figure_dir.mkdir()

    frames: list[pd.DataFrame] = []
    instance_reports: list[dict[str, object]] = []
    representative_done: set[tuple[str, str]] = set()
    schedules = {
        "temperature": schedule_from_dict(profile["temperature_schedule"]),
        "gain": schedule_from_dict(profile["gain_schedule"]),
    }
    run_counter = 0
    instance_seeds = profile["experiment"]["instance_seeds"][: args.instance_limit]
    run_seeds = profile["experiment"]["run_seeds"][: args.run_limit]
    for instance_seed in instance_seeds:
        instance = generate_sk(
            n=n, seed=int(instance_seed), j_scale=float(profile["model"]["j_scale"])
        )
        instance.save(raw_dir / f"sk_n{n}_seed{instance_seed}.npz")
        instance_reports.append(instance.validation())
        reference = (
            brute_force_ground_energy(instance) if n <= 20 else finite_size_ground_energy_mean(n)
        )
        for run_seed in run_seeds:
            for model_kind in model_kinds:
                config = replace(
                    simulation_config(profile, model_kind=model_kind),
                    dt=args.dt or float(profile["integration"]["dt"]),
                    steps=args.steps or int(profile["integration"]["steps"]),
                    record_stride=args.record_stride
                    or int(profile["integration"]["record_stride"]),
                )
                for schedule_name, schedule in schedules.items():
                    run_counter += 1
                    result = simulate(
                        instance,
                        schedule,
                        config,
                        seed=int(run_seed),
                        reference_energy=reference,
                    )
                    run_label = f"i{instance_seed}_r{run_seed}_{model_kind}_{schedule_name}"
                    result.save(raw_dir / f"{run_label}.npz")
                    frames.append(
                        result_frame(
                            result,
                            schedule=schedule_name,
                            model=model_kind,
                            run_id=run_label,
                            instance_seed=instance_seed,
                            run_seed=run_seed,
                            n=n,
                        )
                    )
                    representative_key = (model_kind, schedule_name)
                    if representative_key not in representative_done:
                        plot_diagnostics(
                            result, figure_dir / f"diagnostics_{model_kind}_{schedule_name}.png"
                        )
                        plot_distribution_evolution(
                            result, figure_dir / f"distribution_{model_kind}_{schedule_name}.png"
                        )
                        plot_state_heatmaps(
                            result, figure_dir / f"heatmaps_{model_kind}_{schedule_name}.png"
                        )
                        representative_done.add(representative_key)

    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_csv(run_dir / "trajectories.csv", index=False)
    comparison = trajectories.copy()
    comparison["schedule"] = comparison["model"] + ":" + comparison["schedule"]
    plot_annealing_comparison(comparison, figure_dir / "annealing_comparison.png")
    final = trajectories.loc[trajectories.groupby("run_id")["time"].idxmax()].copy()
    final.to_csv(run_dir / "final_results.csv", index=False)
    aggregate = (
        final.groupby(["model", "schedule"])[["energy", "best_energy", "flip_rate", "p0_kde"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = ["_".join(filter(None, map(str, column))) for column in aggregate.columns]
    aggregate.to_csv(run_dir / "aggregate.csv", index=False)
    paired_rows = []
    for model_kind, group in final.groupby("model"):
        pivot = group.pivot(
            index=["instance_seed", "run_seed"], columns="schedule", values="best_energy"
        ).dropna()
        difference = pivot["temperature"] - pivot["gain"]
        sem95 = (
            1.96 * float(difference.std(ddof=1)) / np.sqrt(len(difference))
            if len(difference) > 1
            else 0.0
        )
        paired_rows.append(
            {
                "model": model_kind,
                "pairs": len(difference),
                "temperature_minus_gain_best_energy_mean": float(difference.mean()),
                "difference_std": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "approximate_95ci_low": float(difference.mean() - sem95),
                "approximate_95ci_high": float(difference.mean() + sem95),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(run_dir / "paired_comparison.csv", index=False)

    details = {
        "n": n,
        "run_count": run_counter,
        "matched_cost": {
            "steps": config.steps,
            "dt": float(profile["integration"]["dt"]),
            "same_instance_and_run_seeds": True,
        },
        "instance_validation": instance_reports,
        "final_summary": aggregate.to_dict(orient="records"),
        "paired_comparison": paired.to_dict(orient="records"),
    }
    save_manifest(run_dir, profile, details)
    print_summary({"run_dir": str(run_dir), "n": n, "run_count": run_counter})


if __name__ == "__main__":
    main()
