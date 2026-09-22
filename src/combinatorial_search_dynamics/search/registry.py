"""Construct configured search algorithms."""

from __future__ import annotations

from typing import Any

from combinatorial_search_dynamics.search.algorithms import (
    RandomizedFirstImprovement,
    ShortTermTabuSearch,
    SimulatedAnnealing,
)


def get_search_algorithm(config: dict[str, Any]) -> object:
    name = str(config["name"])
    if name == "randomized_first_improvement":
        return RandomizedFirstImprovement()
    if name == "simulated_annealing":
        return SimulatedAnnealing(
            initial_temperature=float(config["initial_temperature"]),
            final_temperature=float(config["final_temperature"]),
        )
    if name == "short_term_tabu":
        return ShortTermTabuSearch(tenure=int(config["tenure"]))
    raise KeyError(f"Unknown search method: {name}")


def get_search_algorithms(config: dict[str, Any]) -> list[object]:
    """Construct one method or every method in a paired comparison."""
    if config.get("name") == "classical_comparison":
        methods = config.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ValueError("classical_comparison.methods must be a non-empty list")
        algorithms = [get_search_algorithm(method) for method in methods]
        names = [algorithm.name for algorithm in algorithms]
        if len(names) != len(set(names)):
            raise ValueError("classical comparison method names must be unique")
        return algorithms
    return [get_search_algorithm(config)]
