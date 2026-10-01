from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zhou_aim.dynamics import SimulationResult
from zhou_aim.visualization import plot_distribution_evolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gain_result", type=Path)
    parser.add_argument("temperature_result", type=Path)
    parser.add_argument("--gain-output", type=Path, required=True)
    parser.add_argument("--temperature-output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args()

    results = [SimulationResult.load(args.gain_result), SimulationResult.load(args.temperature_result)]
    bound = max(float(np.quantile(np.abs(result.x), 0.995)) for result in results)
    bound = max(bound, 0.25)
    edges = np.linspace(-bound, bound, args.bins + 1)
    density_vmax = max(
        float(np.histogram(row, bins=edges, density=True)[0].max())
        for result in results
        for row in result.x
    )

    for result, output in zip(
        results, [args.gain_output, args.temperature_output], strict=True
    ):
        plot_distribution_evolution(
            result,
            output,
            bins=args.bins,
            bound=bound,
            density_vmax=density_vmax,
            relative_density=True,
        )


if __name__ == "__main__":
    main()