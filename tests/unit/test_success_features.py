from __future__ import annotations

import math

import pytest

from llm_search_dynamics.evaluation.success_features import (
    evaluate_predictions,
    fit_logistic,
    run_feature_selection,
)
from llm_search_dynamics.features.success import (
    B2_FEATURES,
    build_instance_feature_rows,
    build_success_feature_rows,
    nearest_optimal_distance,
    neighborhood_features,
)
from llm_search_dynamics.tasks.knapsack import KnapsackInstance


def instance() -> KnapsackInstance:
    return KnapsackInstance(
        weights=(1, 2, 3),
        values=(3, 3, 4),
        capacity=3,
        item_count=3,
        generation_seed=7,
        task_name="knapsack",
        task_version="1",
    )


def test_local_features_include_infeasible_neighbors_and_nearest_optimum() -> None:
    measured = neighborhood_features(
        instance(),
        (1, 1, 0),
        objective_scale=6.0,
        epsilon=1.0e-9,
    )
    assert measured["M2_improving_fraction"] == 0.0
    assert measured["M2_best_gain"] == 0.0
    expected_entropy = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert measured["M2_local_entropy"] == pytest.approx(expected_entropy)
    assert nearest_optimal_distance(instance(), (0, 0, 1), 6.0, solve_seed=1) == 1.0


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for instance_index in range(4):
            success = instance_index % 2 == 0
            for checkpoint in range(2):
                signal = 1.0 if success else 0.0
                rows.append(
                    {
                        "checkpoint_id": f"c-{split}-{instance_index}-{checkpoint}",
                        "trial_id": f"t-{split}-{instance_index}",
                        "instance_id": f"i-{split_index}-{instance_index}",
                        "split": split,
                        "search_method_name": "method",
                        "success": success,
                        "trial_weight": 0.5,
                        "remaining_budget_fraction": 1.0 - checkpoint / 2,
                        "relative_objective_gap": 0.5,
                        "M2_distance": signal,
                        "M2_improving_fraction": 0.0,
                        "M2_best_gain": 0.0,
                        "M2_local_entropy": 0.0,
                        "M3_progress_rate": 0.0,
                        "M3_stagnation": 0.0,
                        "M3_revisit": 0.0,
                        "M3_best_update_rate": 0.0,
                        "M3_visit_entropy": 0.0,
                    }
                )
    return rows


def test_weighted_logistic_selection_uses_validation_and_reports_paired_test() -> None:
    rows = _synthetic_rows()
    settings = {
        "regularization": 0.01,
        "max_iterations": 100,
        "tolerance": 1.0e-8,
        "probability_floor": 1.0e-9,
        "calibration_bins": 4,
        "min_delta": 1.0e-5,
        "max_features_m2": 2,
        "max_features_m3": 2,
        "bootstrap_samples": 30,
    }
    models, selection, comparison, predictions = run_feature_selection(
        rows, settings=settings, bootstrap_seed=11
    )
    assert set(models) == {"method"}
    assert "M2_distance" in selection["method"]["selected_m2_features"]
    assert comparison["method"]["paired_vs_B2"]["brier_improvement"] > 0
    assert len(predictions) == 8
    assert {row["instance_id"].split("-")[1] for row in predictions} == {"2"}


def test_logistic_metrics_are_finite_and_instance_macro() -> None:
    rows = [row for row in _synthetic_rows() if row["split"] == "train"]
    model = fit_logistic(
        rows,
        B2_FEATURES,
        regularization=1.0,
        max_iterations=100,
        tolerance=1.0e-8,
    )
    probabilities = model.predict(rows)
    metrics = evaluate_predictions(
        rows, probabilities, probability_floor=1.0e-9, calibration_bins=4
    )
    assert 0 <= metrics["brier"] <= 1
    assert math.isfinite(float(metrics["nll"]))


def test_instance_sampling_features_are_reproducible() -> None:
    source = {
        "instance_id": "ins-test",
        "instance_json": '{"capacity":3,"generation_seed":7,"item_count":3,"task_name":"knapsack","task_version":"1","values":[3,3,4],"weights":[1,2,3]}',
    }
    reference = {
        "instance_id": "ins-test",
        "optimal_value": 6.0,
    }
    arguments = {
        "instances": [source],
        "references": [reference],
        "sample_seed": 10,
        "sample_count": 8,
        "random_walk_steps": 8,
        "epsilon": 1.0e-9,
    }
    assert build_instance_feature_rows(**arguments) == build_instance_feature_rows(**arguments)


def test_history_features_use_only_current_and_past_checkpoints() -> None:
    source = {
        "instance_id": "ins-test",
        "instance_json": '{"capacity":3,"generation_seed":7,"item_count":3,"task_name":"knapsack","task_version":"1","values":[3,3,4],"weights":[1,2,3]}',
    }
    trial = {
        "trial_id": "trial",
        "instance_id": "ins-test",
        "search_method_name": "method",
        "budget_limit": 10,
        "success": False,
    }
    points = [
        {
            "checkpoint_id": "c0",
            "trial_id": "trial",
            "checkpoint_index": 0,
            "budget_used": 0,
            "remaining_budget": 10,
            "objective_value": 0.0,
            "optimality_gap": 1.0,
            "is_terminal": False,
            "state_json": '{"selected":[0,0,0]}',
        },
        {
            "checkpoint_id": "c1",
            "trial_id": "trial",
            "checkpoint_index": 1,
            "budget_used": 2,
            "remaining_budget": 8,
            "objective_value": 3.0,
            "optimality_gap": 0.5,
            "is_terminal": False,
            "state_json": '{"selected":[1,0,0]}',
        },
        {
            "checkpoint_id": "c2",
            "trial_id": "trial",
            "checkpoint_index": 2,
            "budget_used": 4,
            "remaining_budget": 6,
            "objective_value": 0.0,
            "optimality_gap": 1.0,
            "is_terminal": False,
            "state_json": '{"selected":[0,0,0]}',
        },
        {
            "checkpoint_id": "c3",
            "trial_id": "trial",
            "checkpoint_index": 3,
            "budget_used": 5,
            "remaining_budget": 5,
            "objective_value": 4.0,
            "optimality_gap": 1 / 3,
            "is_terminal": True,
            "state_json": '{"selected":[0,0,1]}',
        },
    ]
    arguments = {
        "instances": [source],
        "trials": [trial],
        "references": [
            {"instance_id": "ins-test", "optimality_proven": True, "optimal_value": 6.0}
        ],
        "assignments": {"ins-test": "train"},
        "history_window_budget": 10,
        "epsilon": 1.0e-9,
        "distance_seed": 1,
    }
    rows = build_success_feature_rows(checkpoints=points, **arguments)
    assert [row["checkpoint_id"] for row in rows] == ["c0", "c1", "c2"]
    assert rows[0]["M3_progress_rate"] == 0.0
    assert rows[0]["M3_stagnation"] == 0.0
    assert rows[2]["M3_revisit"] == pytest.approx(0.5)
    without_future = build_success_feature_rows(checkpoints=points[:-1], **arguments)
    assert rows == without_future
