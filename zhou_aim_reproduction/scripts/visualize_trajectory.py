from __future__ import annotations

import argparse
from pathlib import Path

from zhou_aim.dynamics import SimulationResult
from zhou_aim.exact_solver import enumerate_landscape
from zhou_aim.sk_instance import SKInstance
from zhou_aim.visualization import (
    animate_diagnostics,
    plot_diagnostics,
    plot_distribution_evolution,
    plot_pca_trajectory,
    plot_small_landscape,
    plot_state_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = SimulationResult.load(args.result)
    plot_diagnostics(result, args.output_dir / "diagnostics.png")
    plot_distribution_evolution(result, args.output_dir / "distribution_evolution.png")
    plot_state_heatmaps(result, args.output_dir / "state_heatmaps.png")
    plot_pca_trajectory(result, args.output_dir / "pca_trajectory.png")
    if args.instance is not None:
        instance = SKInstance.load(args.instance)
        if instance.n <= 22:
            plot_small_landscape(
                enumerate_landscape(instance), result, args.output_dir / "exact_landscape.png"
            )
    if args.animate:
        animate_diagnostics(result, args.output_dir / "trajectory.gif")


if __name__ == "__main__":
    main()
