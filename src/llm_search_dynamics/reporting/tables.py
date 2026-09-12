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
    """Write the Phase 2 report and a machine-readable two-column summary."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(key, value) for key, value in summary.items()]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    artifacts = "\n".join(f"- `{path.as_posix()}`" for path in artifact_paths)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 2 CPU pilot report",
                "",
                "This run validates the pipeline. It is not a research result.",
                "",
                "## Run",
                "",
                f"- MLflow run ID: `{summary['run_id']}`",
                f"- Config hash: `{summary['config_hash']}`",
                f"- Task: `{summary['task']}`",
                f"- Generator: `{summary['generator']}` (not an LLM adapter)",
                (
                    f"- Instances / trials / checkpoints: {summary['instances']} / "
                    f"{summary['trials']} / {summary['checkpoints']}"
                ),
                (
                    f"- Train / validation / test instances: {summary['train_instances']} / "
                    f"{summary['validation_instances']} / {summary['test_instances']}"
                ),
                f"- Success rate: {summary['success_rate']:.6f}",
                f"- Mean final Hamming distance: {summary['mean_final_hamming_distance']:.6f}",
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
                "- State representation: Hamming distance; state 0 is absorbing success.",
                "- Test instances were excluded from state-model and transition-model fitting.",
                "",
                "## Artifacts",
                "",
                artifacts,
                "",
                "## Deferred work",
                "",
                (
                    "Real tasks, solvers, LLM adapters, prompt versioning, internal observations, "
                    "Optuna, nested cross-validation, and confirmatory evaluation remain for Phase 3+."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path, summary_path
