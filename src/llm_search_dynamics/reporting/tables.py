"""Small Markdown and CSV reports that require no rendering service."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def build_pilot_report(
    *,
    report_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
    artifact_paths: list[Path],
) -> tuple[Path, Path]:
    """Write a Phase 2/3 report and a machine-readable two-column summary."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(key, value) for key, value in summary.items()]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    artifacts = "\n".join(f"- `{path.as_posix()}`" for path in artifact_paths)
    problem_section = (
        [
            (
                "- Item count / weight range / value range / capacity ratio: "
                f"{summary['item_count']} / {summary['weight_range']} / "
                f"{summary['value_range']} / {summary['capacity_ratio']}"
            ),
            f"- OR-Tools CP-SAT solver: `{summary['solver']}`; status counts: {summary['solver_status_counts']}",
            (
                f"- Solver timeouts: {summary['solver_timeout_count']}; proven-optimal rate: "
                f"{summary['solver_optimality_proven_rate']:.6f}"
            ),
            f"- Mean solver runtime (seconds): {summary['solver_runtime_mean_seconds']:.6f}",
            f"- Mean final total value: {summary['mean_final_total_value']:.6f}",
            (
                f"- Test trial / instance success rate: {summary['knapsack_success_rate_trial']:.6f} / "
                f"{summary['knapsack_success_rate_instance']:.6f}"
            ),
            (
                "- Test final absolute / relative gap: "
                f"{summary.get('knapsack_final_absolute_gap_mean', 'unavailable')} / "
                f"{summary.get('knapsack_final_relative_gap_mean', 'unavailable')}"
            ),
            (
                "- Best-so-far value / evaluations used to reach optimum: "
                f"{summary['knapsack_best_so_far_value_mean']} / "
                f"{summary.get('knapsack_optimal_reached_budget_mean', 'unavailable')}"
            ),
            f"- Feasible checkpoint rate: {summary['knapsack_feasible_checkpoint_rate']:.6f}",
            (
                "- Unproven references are excluded from gap-state fit and Markov test metrics; "
                "timeouts remain in solver denominators."
            ),
        ]
        if summary["problem_type"] == "knapsack"
        else [f"- Mean final Hamming distance: {summary['mean_final_hamming_distance']:.6f}"]
    )
    report_path.write_text(
        "\n".join(
            [
                f"# Phase {3 if summary['problem_type'] == 'knapsack' else 2} CPU pilot report",
                "",
                "This run validates the pipeline. It is not a research result.",
                "",
                "## Run",
                "",
                f"- MLflow run ID: `{summary['run_id']}`",
                f"- Config hash: `{summary['config_hash']}`",
                f"- Task: `{summary['task']}`",
                f"- Search method: `{summary['search_method']}`",
                (
                    f"- Instances / trials / checkpoints: {summary['instances']} / "
                    f"{summary['trials']} / {summary['checkpoints']}"
                ),
                (
                    f"- Train / validation / test instances: {summary['train_instances']} / "
                    f"{summary['validation_instances']} / {summary['test_instances']}"
                ),
                f"- Success rate: {summary['success_rate']:.6f}",
                *problem_section,
                "",
                "## Held-out evaluation",
                "",
                f"- One-step negative log-likelihood: {summary['one_step_nll']:.6f}",
                f"- One-step next-state accuracy: {summary['one_step_accuracy']:.6f}",
                f"- Success-state Brier score: {summary['success_brier_score']:.6f}",
                (
                    "- Evaluated transitions / instances / trials: "
                    f"{summary['evaluated_transitions']} / {summary['evaluated_instances']} / "
                    f"{summary['evaluated_trials']}"
                ),
                "- State representation: "
                + (
                    "proven-optimum relative objective-gap bins"
                    if summary["problem_type"] == "knapsack"
                    else "Hamming distance"
                )
                + "; state 0 is absorbing success.",
                "- Test instances were excluded from state-model and transition-model fitting.",
                "",
                "## Artifacts",
                "",
                artifacts,
                "",
                "## Deferred work",
                "",
                (
                    "LLM adapters, prompt versioning, internal observations, interruption recovery, "
                    "Optuna, nested cross-validation, and confirmatory evaluation remain for Phase 4+."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path, summary_path
