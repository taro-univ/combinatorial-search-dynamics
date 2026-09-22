"""One-step predictions from a fitted transition model."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from combinatorial_search_dynamics.dynamics.transition import MarkovTransitionModel


def predict_one_step(
    model: MarkovTransitionModel, trajectories: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return compact, checkpoint-pair predictions suitable for re-aggregation."""
    predictions: list[dict[str, Any]] = []
    for trajectory in trajectories:
        states = [int(value) for value in trajectory["states"]]
        checkpoint_ids = trajectory["checkpoint_ids"]
        for index, (current, observed) in enumerate(pairwise(states)):
            probabilities = model.predict_distribution(current, 1)
            predictions.append(
                {
                    "instance_id": trajectory["instance_id"],
                    "trial_id": trajectory["trial_id"],
                    "from_checkpoint_id": checkpoint_ids[index],
                    "to_checkpoint_id": checkpoint_ids[index + 1],
                    "current_state": current,
                    "observed_next_state": observed,
                    "predicted_next_state": int(probabilities.argmax()),
                    "observed_probability": float(probabilities[observed]),
                    "success_probability": float(probabilities[0]),
                }
            )
    return predictions
