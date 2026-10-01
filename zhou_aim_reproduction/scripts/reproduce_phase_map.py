from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from common import load_profile, print_summary, save_manifest, simulation_config, unique_run_dir

from zhou_aim.dynamics import simulate
from zhou_aim.observables import finite_size_ground_energy_mean
from zhou_aim.schedules import LinearSchedule, schedule_from_dict
from zhou_aim.sk_instance import generate_sk
from zhou_aim.visualization import plot_phase_map


def _freeze_row(result, n: int) -> dict[str, float | bool]:
    threshold = 1.0 / (n * result.time[-1])
    terminal_quiet = result.cumulative_flips == result.cumulative_flips[-1]
    candidates = np.flatnonzero(terminal_quiet & (result.flip_rate < threshold))
    freeze_index = int(candidates[0]) if len(candidates) else len(result.time) - 1
    quiet_duration = result.time[-1] - result.time[freeze_index]
    minimum_quiet_duration = max(
        0.1 * result.time[-1],
        result.config.flip_window_steps * result.config.dt,
    )
    frozen = bool(quiet_duration >= minimum_quiet_duration and result.flip_rate[-1] < threshold)
    if not frozen:
        freeze_index = len(result.time) - 1
    return {
        "frozen": frozen,
        "threshold": threshold,
        "minimum_quiet_duration": minimum_quiet_duration,
        "terminal_quiet_duration": float(quiet_duration),
        "freeze_time": float(result.time[freeze_index]),
        "freeze_gain": float(result.gain[freeze_index]),
        "freeze_temperature": float(result.temperature[freeze_index]),
        "freeze_energy": float(result.energy[freeze_index]),
        "freeze_flip_rate": float(result.flip_rate[freeze_index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--run-id")
    parser.add_argument("--model-kind", choices=["cim", "digcim"], default="cim")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    phase = profile["phase_map"]
    n = int(phase["n"])
    run_dir = unique_run_dir("phase_map", args.run_id)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir()

    base = simulation_config(profile, model_kind=args.model_kind)
    config = replace(
        base,
        steps=int(phase["burn_steps"]) + int(phase["sample_steps"]),
        record_stride=max(1, base.record_stride),
    )
    tail_time = float(phase["sample_steps"]) * config.dt
    rows: list[dict[str, float | int]] = []
    for gain in phase["a_values"]:
        for temperature in phase["t_values"]:
            per_seed = []
            for seed in phase["seeds"]:
                instance = generate_sk(n, int(seed), float(profile["model"]["j_scale"]))
                schedule = LinearSchedule(gain, gain, temperature, temperature, name="fixed")
                result = simulate(
                    instance,
                    schedule,
                    config,
                    seed=100000 + int(seed),
                    reference_energy=finite_size_ground_energy_mean(n),
                )
                mask = result.time >= result.time[-1] - tail_time
                per_seed.append(
                    {
                        "energy": float(np.mean(result.energy[mask])),
                        "p0": float(np.mean(result.p0_kde[mask])),
                        "p0_hist": float(np.mean(result.p0_hist[mask])),
                        "flip_rate": float(np.mean(result.flip_rate[mask])),
                    }
                )
            rows.append(
                {
                    "gain": float(gain),
                    "temperature": float(temperature),
                    "energy_mean": float(np.mean([row["energy"] for row in per_seed])),
                    "energy_std": float(np.std([row["energy"] for row in per_seed], ddof=1)),
                    "p0_mean": float(np.mean([row["p0"] for row in per_seed])),
                    "p0_std": float(np.std([row["p0"] for row in per_seed], ddof=1)),
                    "p0_hist_mean": float(np.mean([row["p0_hist"] for row in per_seed])),
                    "p0_hist_std": float(
                        np.std([row["p0_hist"] for row in per_seed], ddof=1)
                    ),
                    "flip_rate_mean": float(np.mean([row["flip_rate"] for row in per_seed])),
                    "flip_rate_std": float(np.std([row["flip_rate"] for row in per_seed], ddof=1)),
                    "replicates": len(per_seed),
                    "active": bool(
                        np.mean([row["flip_rate"] for row in per_seed]) >= 1.0 / (n * tail_time)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "phase_map.csv", index=False)

    path_results = {}
    plotted_paths = {}
    freeze_points = {}
    path_instance = generate_sk(n, 991, float(profile["model"]["j_scale"]))
    for name, key in (("temperature", "temperature_schedule"), ("gain", "gain_schedule")):
        schedule = schedule_from_dict(profile[key])
        result = simulate(
            path_instance,
            schedule,
            config,
            seed=1771,
            reference_energy=finite_size_ground_energy_mean(n),
        )
        diagnosis = _freeze_row(result, n)
        path_results[name] = diagnosis
        plotted_paths[name] = np.column_stack([result.gain, result.temperature])
        freeze_points[name] = (diagnosis["freeze_gain"], diagnosis["freeze_temperature"])
    pd.DataFrame.from_dict(path_results, orient="index").rename_axis("path").reset_index().to_csv(
        run_dir / "path_freezing.csv", index=False
    )
    plot_phase_map(
        frame,
        plotted_paths,
        figure_dir / "phase_map.png",
        freeze_points=freeze_points,
    )

    details = {
        "model": args.model_kind,
        "n": n,
        "grid_points": len(frame),
        "replicates_per_point": len(phase["seeds"]),
        "finite_time_direct_simulation": True,
        "path_freezing": path_results,
    }
    save_manifest(run_dir, profile, details)
    print_summary({"run_dir": str(run_dir), **details})


if __name__ == "__main__":
    main()
