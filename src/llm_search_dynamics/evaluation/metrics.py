"""Phase 2 one-step probabilistic metrics and their aggregation units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_search_dynamics.dynamics.transition import MarkovTransitionModel
from llm_search_dynamics.evaluation.prediction import predict_one_step


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    n_transitions: int
    n_instances: int
    n_trials: int
    predictions: tuple[dict[str, Any], ...]
    per_instance: tuple[dict[str, Any], ...]


def evaluate_trajectories(
    model: MarkovTransitionModel,
    trajectories: list[dict[str, Any]],
    *,
    probability_floor: float = 1e-12,
) -> EvaluationResult:
    """Evaluate one-step NLL, accuracy, and success-event Brier score.

    NLL clips the observed probability at ``probability_floor``. Brier score is
    the squared error for predicting whether the next state is absorbing state 0.
    Each transition is one unit for these three metrics.
    """
    if not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be between 0 and 1")
    predictions = predict_one_step(model, trajectories)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        grouped.setdefault(str(prediction["instance_id"]), []).append(prediction)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {
                "one_step_nll": 0.0,
                "one_step_accuracy": 0.0,
                "success_brier_score": 0.0,
            }
        return {
            "one_step_nll": float(
                np.mean(
                    [-np.log(max(row["observed_probability"], probability_floor)) for row in rows]
                )
            ),
            "one_step_accuracy": float(
                np.mean([row["predicted_next_state"] == row["observed_next_state"] for row in rows])
            ),
            "success_brier_score": float(
                np.mean(
                    [
                        (row["success_probability"] - (row["observed_next_state"] == 0)) ** 2
                        for row in rows
                    ]
                )
            ),
        }

    per_instance = tuple(
        {
            "instance_id": instance_id,
            "n_transitions": len(rows),
            **aggregate(rows),
        }
        for instance_id, rows in sorted(grouped.items())
    )
    return EvaluationResult(
        metrics=aggregate(predictions),
        n_transitions=len(predictions),
        n_instances=len({str(row["instance_id"]) for row in trajectories}),
        n_trials=len({str(row["trial_id"]) for row in trajectories}),
        predictions=tuple(predictions),
        per_instance=per_instance,
    )
