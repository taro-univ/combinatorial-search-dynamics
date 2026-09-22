"""Regularized logistic comparisons and validation-only feature selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from combinatorial_search_dynamics.features.success import B2_FEATURES, M2_FEATURES, M3_FEATURES


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    converged: bool
    iterations: int

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in rows], dtype=float
        )
        if matrix.size:
            matrix = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        design = np.column_stack([np.ones(len(rows)), matrix])
        logits = np.clip(design @ np.asarray(self.coefficients), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "converged": self.converged,
            "iterations": self.iterations,
        }


def fit_logistic(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    regularization: float,
    max_iterations: int,
    tolerance: float,
) -> LogisticModel:
    """Fit a deterministic weighted L2 logistic regression using Newton updates."""
    if not rows:
        raise ValueError("Cannot fit logistic regression without rows")
    if regularization < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Invalid logistic fitting settings")
    raw = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    labels = np.asarray([float(bool(row["success"])) for row in rows])
    weights = np.asarray([float(row["trial_weight"]) for row in rows])
    total_weight = float(weights.sum())
    means = np.average(raw, axis=0, weights=weights)
    variance = np.average((raw - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(variance)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(rows)), (raw - means) / scales])
    coefficients = np.zeros(design.shape[1], dtype=float)
    prevalence = (float(weights @ labels) + 0.5) / (total_weight + 1.0)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.diag([0.0, *([regularization] * len(feature_names))])
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (weights * (probabilities - labels)) / total_weight
        gradient += penalty @ coefficients
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / total_weight + penalty
        update = np.linalg.pinv(hessian) @ gradient
        coefficients -= update
        if float(np.max(np.abs(update))) < tolerance:
            converged = True
            break
    return LogisticModel(
        feature_names=feature_names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        converged=converged,
        iterations=iterations,
    )


def _auc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = [value for value, label in zip(probabilities, labels, strict=True) if label]
    negatives = [value for value, label in zip(probabilities, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    score = 0.0
    for positive in positives:
        for negative in negatives:
            score += float(positive > negative) + 0.5 * float(positive == negative)
    return score / (len(positives) * len(negatives))


def evaluate_predictions(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    probability_floor: float,
    calibration_bins: int,
) -> dict[str, float | None]:
    """Evaluate losses by instance before averaging across instances."""
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("Rows and probabilities must have equal positive length")
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["instance_id"]), []).append(index)
    brier_by_instance: list[float] = []
    nll_by_instance: list[float] = []
    labels = np.asarray([float(bool(row["success"])) for row in rows])
    clipped = np.clip(probabilities, probability_floor, 1.0 - probability_floor)
    for indexes in grouped.values():
        y = labels[indexes]
        p = probabilities[indexes]
        pc = clipped[indexes]
        brier_by_instance.append(float(np.mean((p - y) ** 2)))
        nll_by_instance.append(float(np.mean(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))))
    calibration_error = 0.0
    for bin_index in range(calibration_bins):
        lower = bin_index / calibration_bins
        upper = (bin_index + 1) / calibration_bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if bin_index == calibration_bins - 1 else probabilities < upper
        )
        if np.any(mask):
            calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    integer_labels = [int(value) for value in labels]
    return {
        "brier": float(np.mean(brier_by_instance)),
        "nll": float(np.mean(nll_by_instance)),
        "auroc": _auc(integer_labels, [float(value) for value in probabilities]),
        "calibration_error": calibration_error,
        "checkpoint_count": float(len(rows)),
        "instance_count": float(len(grouped)),
    }


def _fit_and_evaluate(
    train_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    features: tuple[str, ...],
    settings: dict[str, Any],
) -> tuple[LogisticModel, dict[str, float | None], np.ndarray]:
    model = fit_logistic(
        train_rows,
        features,
        regularization=float(settings["regularization"]),
        max_iterations=int(settings["max_iterations"]),
        tolerance=float(settings["tolerance"]),
    )
    probabilities = model.predict(evaluation_rows)
    metrics = evaluate_predictions(
        evaluation_rows,
        probabilities,
        probability_floor=float(settings["probability_floor"]),
        calibration_bins=int(settings["calibration_bins"]),
    )
    return model, metrics, probabilities


def _forward_select(
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    starting: tuple[str, ...],
    candidates: tuple[str, ...],
    max_features: int,
    settings: dict[str, Any],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    selected = list(starting)
    remaining = list(candidates)
    _, baseline_metrics, _ = _fit_and_evaluate(
        train_rows, validation_rows, tuple(selected), settings
    )
    current = float(baseline_metrics["brier"])
    history: list[dict[str, Any]] = []
    for step in range(max_features):
        attempts: list[tuple[float, str, dict[str, float | None]]] = []
        for candidate in remaining:
            _, metrics, _ = _fit_and_evaluate(
                train_rows, validation_rows, (*selected, candidate), settings
            )
            attempts.append((float(metrics["brier"]), candidate, metrics))
        if not attempts:
            break
        score, candidate, metrics = min(attempts, key=lambda item: (item[0], item[1]))
        improvement = current - score
        accepted = improvement >= float(settings["min_delta"])
        history.append(
            {
                "step": step + 1,
                "candidate": candidate,
                "validation_metrics": metrics,
                "brier_improvement": improvement,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        current = score
    return tuple(selected), history


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    final: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_instance: dict[str, list[int]] = {}
    labels = np.asarray([float(bool(row["success"])) for row in rows])
    for index, row in enumerate(rows):
        by_instance.setdefault(str(row["instance_id"]), []).append(index)
    identifiers = sorted(by_instance)
    differences = {
        identifier: float(
            np.mean((baseline[by_instance[identifier]] - labels[by_instance[identifier]]) ** 2)
            - np.mean((final[by_instance[identifier]] - labels[by_instance[identifier]]) ** 2)
        )
        for identifier in identifiers
    }
    rng = random.Random(seed)
    estimates = [
        float(np.mean([differences[rng.choice(identifiers)] for _ in identifiers]))
        for _ in range(samples)
    ]
    return {
        "brier_improvement": float(np.mean(list(differences.values()))),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
    }


def run_feature_selection(
    rows: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Select features independently per search method and evaluate test once."""
    if {bool(row["success"]) for row in rows} != {False, True}:
        raise ValueError("Success prediction requires both successful and failed trials")
    models: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    methods = sorted({str(row["search_method_name"]) for row in rows})
    for method_index, method in enumerate(methods):
        method_rows = [row for row in rows if row["search_method_name"] == method]
        train = [row for row in method_rows if row["split"] == "train"]
        validation = [row for row in method_rows if row["split"] == "validation"]
        test = [row for row in method_rows if row["split"] == "test"]
        if not train or not validation or not test:
            raise ValueError(f"Search method {method} has an empty split")
        singles: dict[str, Any] = {}
        single_models: dict[str, Any] = {}
        for candidate in (*M2_FEATURES, *M3_FEATURES):
            single_model, metrics, _ = _fit_and_evaluate(
                train, validation, (*B2_FEATURES, candidate), settings
            )
            singles[candidate] = {
                "features": [*B2_FEATURES, candidate],
                "validation_metrics": metrics,
            }
            single_models[candidate] = single_model.to_dict()
        selected_m2, m2_history = _forward_select(
            train_rows=train,
            validation_rows=validation,
            starting=B2_FEATURES,
            candidates=M2_FEATURES,
            max_features=int(settings["max_features_m2"]),
            settings=settings,
        )
        selected_final, m3_history = _forward_select(
            train_rows=train,
            validation_rows=validation,
            starting=selected_m2,
            candidates=M3_FEATURES,
            max_features=int(settings["max_features_m3"]),
            settings=settings,
        )
        configurations = {
            "B2": B2_FEATURES,
            "selected_M2": selected_m2,
            "selected_final": selected_final,
        }
        method_models: dict[str, Any] = {}
        method_comparison: dict[str, Any] = {}
        probabilities: dict[str, np.ndarray] = {}
        for name, features in configurations.items():
            model, metrics, predicted = _fit_and_evaluate(train, test, features, settings)
            method_models[name] = model.to_dict()
            method_comparison[name] = {"features": list(features), "test_metrics": metrics}
            probabilities[name] = predicted
        paired = _paired_bootstrap(
            test,
            probabilities["B2"],
            probabilities["selected_final"],
            samples=int(settings["bootstrap_samples"]),
            seed=bootstrap_seed + method_index,
        )
        test_trials = {
            str(row["trial_id"]): (str(row["instance_id"]), bool(row["success"])) for row in test
        }
        test_instances: dict[str, bool] = {}
        for instance_id, success in test_trials.values():
            test_instances[instance_id] = test_instances.get(instance_id, False) or success
        method_comparison["outcomes"] = {
            "trial_success_rate": sum(success for _, success in test_trials.values())
            / len(test_trials),
            "instance_success_rate": sum(test_instances.values()) / len(test_instances),
            "trial_count": len(test_trials),
            "instance_count": len(test_instances),
        }
        for row_index, row in enumerate(test):
            prediction_rows.append(
                {
                    "analysis_version": "1",
                    "checkpoint_id": row["checkpoint_id"],
                    "trial_id": row["trial_id"],
                    "instance_id": row["instance_id"],
                    "search_method_name": method,
                    "success": row["success"],
                    "B2_probability": float(probabilities["B2"][row_index]),
                    "selected_M2_probability": float(probabilities["selected_M2"][row_index]),
                    "selected_final_probability": float(probabilities["selected_final"][row_index]),
                }
            )
        models[method] = {**method_models, "single_additions": single_models}
        selections[method] = {
            "single_additions": singles,
            "m2_forward_history": m2_history,
            "m3_forward_history": m3_history,
            "selected_m2_features": list(selected_m2),
            "selected_final_features": list(selected_final),
        }
        comparisons[method] = {"models": method_comparison, "paired_vs_B2": paired}
    return models, selections, comparisons, prediction_rows
