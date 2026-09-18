"""Static checks for the local Phase 3 DVC graph."""

from __future__ import annotations

import yaml

from llm_search_dynamics.config import repository_root


def test_dvc_graph_is_local_acyclic_and_stage_complete() -> None:
    root = repository_root()
    stages = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    assert list(stages) == [
        "generate_instances",
        "solve_references",
        "collect_trajectories",
        "prepare_trajectories",
        "extract_features",
        "make_splits",
        "fit_model",
        "evaluate_model",
        "build_report",
    ]
    output_owner = {out: stage for stage, spec in stages.items() for out in spec["outs"]}
    assert len(output_owner) == sum(len(spec["outs"]) for spec in stages.values())
    seen = set()
    for stage, spec in stages.items():
        assert spec["cmd"].startswith("uv run lsd ")
        assert spec["deps"] and spec["outs"]
        assert any(dep.startswith("configs/") for dep in spec["deps"]) or stage in {
            "prepare_trajectories",
            "build_report",
        }
        assert all(output_owner[dep] in seen for dep in spec["deps"] if dep in output_owner)
        seen.add(stage)
    assert (root / ".dvc" / "config").is_file()
    dvc_config = (root / ".dvc" / "config").read_text(encoding="utf-8")
    assert "remote" not in dvc_config
