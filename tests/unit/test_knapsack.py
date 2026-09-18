"""Task, objective direction, CP-SAT and reference-table contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ortools.sat.python import cp_model

from llm_search_dynamics.data.parquet import read_parquet, write_parquet
from llm_search_dynamics.data.reference_validation import validate_references
from llm_search_dynamics.data.references import reference_row
from llm_search_dynamics.data.schemas import SCHEMA_VERSION, get_schema
from llm_search_dynamics.generation.mock import MockGenerator
from llm_search_dynamics.identifiers import canonical_json, instance_id
from llm_search_dynamics.solvers.base import SolverParameters, SolverStatus
from llm_search_dynamics.solvers.ortools_knapsack import (
    OrtoolsKnapsackSolver,
    limit_reached,
    mapped_status,
    relative_maximization_gap,
)
from llm_search_dynamics.solvers.registry import get_solver
from llm_search_dynamics.tasks.dummy_binary import BinaryInstance, DummyBinaryTask
from llm_search_dynamics.tasks.knapsack import KnapsackInstance, KnapsackTask
from llm_search_dynamics.tasks.registry import get_task


def task() -> KnapsackTask:
    return KnapsackTask(
        item_count=3,
        min_weight=1,
        max_weight=5,
        min_value=1,
        max_value=10,
        capacity_ratio=0.5,
        max_steps=4,
    )


def fixed_instance() -> KnapsackInstance:
    return KnapsackInstance((2, 3, 4), (4, 5, 6), 5, 3, 17)


def parameters() -> SolverParameters:
    return SolverParameters(5.0, 1, 11, 1e-9, False)


def test_knapsack_generation_state_and_toggle_contract() -> None:
    selected = task()
    assert selected.generate_instance(10) == selected.generate_instance(10)
    assert selected.generate_instance(10) != selected.generate_instance(11)
    generated = selected.generate_instance(10)
    assert len(generated.weights) == len(generated.values) == generated.item_count
    assert generated.capacity < sum(generated.weights)
    state = selected.initial_state(fixed_instance())
    assert state.selected == (0, 0, 0) and selected.is_feasible(state)
    first = selected.apply_action(state, 0)
    assert state.selected == (0, 0, 0)
    assert first.total_weight == 2 and first.total_value == 4
    assert 2 not in selected.legal_actions(first)
    with pytest.raises(ValueError, match="capacity"):
        selected.apply_action(first, 2)
    second = selected.apply_action(first, 1)
    assert selected.objective(second) == 9.0
    assert selected.apply_action(second, 0).selected == (0, 1, 0)
    assert selected.deserialize_state(selected.serialize_state(second)) == second
    assert selected.parse_action("1") == 1
    for invalid in ("-1", "1.0", "item 1", "3"):
        with pytest.raises(ValueError):
            selected.parse_action(invalid)
    with pytest.raises(ValueError):
        selected.apply_action(state, 3)
    assert not selected.is_feasible(replace(second, capacity=4))
    with pytest.raises(ValueError):
        selected.validate_instance(replace(fixed_instance(), values=(1,)))
    with pytest.raises(TypeError):
        task().generate_instance("bad")
    with pytest.raises(KeyError):
        get_task("unknown", max_steps=4)
    assert (
        get_task(
            "knapsack",
            item_count=3,
            min_weight=1,
            max_weight=5,
            min_value=1,
            max_value=10,
            capacity_ratio=0.5,
            max_steps=4,
        ).name
        == "knapsack"
    )


def test_objective_direction_and_mock_reference_separation() -> None:
    binary = DummyBinaryTask(bit_count=2, max_steps=2)
    assert binary.is_better(0, 1)
    knapsack = task()
    assert knapsack.is_better(5, 4)
    initial = knapsack.initial_state(fixed_instance())
    assert not knapsack.is_success(initial)
    optimal = knapsack.apply_action(knapsack.apply_action(initial, 0), 1)
    assert not knapsack.is_success(optimal)
    assert knapsack.is_success(optimal, 9.0)
    greedy = MockGenerator(greedy_probability=1.0, max_steps=1)
    result = greedy.generate(
        task=knapsack,
        instance=fixed_instance(),
        source_instance_id="ins_fixed",
        experiment_conditions={"test": True},
        sampling_seed=2,
        reference_value=9.0,
    )
    assert result.checkpoints[1].state.selected == (0, 0, 1)
    assert not result.success  # greedy receives no reference solution action sequence
    dummy = greedy.generate(
        task=binary,
        instance=BinaryInstance(2, (1, 0), (0, 0)),
        source_instance_id="ins_dummy",
        experiment_conditions={"test": True},
        sampling_seed=2,
    )
    assert dummy.checkpoints[1].objective == 0.0


def test_cp_sat_optimum_gap_status_and_revalidation() -> None:
    solver = OrtoolsKnapsackSolver(task=task(), parameters=parameters())
    result = solver.solve(fixed_instance(), instance_id="ins_fixed")
    assert result.status == SolverStatus.OPTIMAL
    assert result.optimality_proven and not result.timed_out
    assert result.best_feasible_value == result.optimal_value == 9.0
    assert result.best_bound == pytest.approx(9.0)
    assert result.optimality_gap == 0.0
    assert result.solution == (1, 1, 0)
    assert result.parameters.num_search_workers == 1 and result.solver_version
    assert relative_maximization_gap(8, 10, epsilon=1e-9) == pytest.approx(0.25)
    assert relative_maximization_gap(0, 1, epsilon=1e-6) == pytest.approx(1e6)
    assert relative_maximization_gap(None, 1, epsilon=1e-9) is None
    assert relative_maximization_gap(10, 9.999999, epsilon=1e-9) == 0.0
    assert limit_reached(SolverStatus.FEASIBLE, runtime=0.96, time_limit=1)
    assert not limit_reached(SolverStatus.OPTIMAL, runtime=1, time_limit=1)
    assert mapped_status(cp_model.FEASIBLE) == SolverStatus.FEASIBLE
    assert mapped_status(cp_model.UNKNOWN) == SolverStatus.UNKNOWN
    assert mapped_status(cp_model.MODEL_INVALID) == SolverStatus.MODEL_INVALID
    with pytest.raises(ValueError):
        mapped_status(-100)
    with pytest.raises(ValueError, match="finite"):
        SolverParameters(float("nan"), 1, 1, 1e-9, False)
    with pytest.raises(KeyError):
        get_solver("unknown", task=task(), parameters=parameters())
    with pytest.raises(ValueError):
        solver.solve(replace(fixed_instance(), weights=(-1, 3, 4)), instance_id="ins_fixed")


def _instance_table(identifier: str) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "instance_id": identifier,
                "task_name": "knapsack",
                "task_version": "1",
                "problem_size": 3,
                "difficulty_value": None,
                "generation_seed": 17,
                "instance_json": canonical_json(fixed_instance().to_dict()),
                "created_at": datetime(2000, 1, 1, tzinfo=UTC),
            }
        ],
        schema=get_schema("instances"),
    )


def test_reference_schema_io_and_semantic_validation(tmp_path) -> None:
    identifier = instance_id(fixed_instance().to_dict())
    result = OrtoolsKnapsackSolver(task=task(), parameters=parameters()).solve(
        fixed_instance(), instance_id=identifier
    )
    row = reference_row(result, task_name="knapsack", task_version="1")
    table = pa.Table.from_pylist([row], schema=get_schema("reference_solutions"))
    path = tmp_path / "reference_solutions.parquet"
    write_parquet(table, path, "reference_solutions")
    assert read_parquet(path, "reference_solutions").equals(table)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
    assert table.schema.metadata[b"logical_table"] == b"reference_solutions"
    with pytest.raises(FileExistsError):
        write_parquet(table, path, "reference_solutions")
    issues = []
    validate_references(table, _instance_table(identifier), issues)
    assert issues == []
    cases = (
        ([row, row], "reference_duplicate_key"),
        (
            [row, {**row, "solver_version": "other", "reference_id": "art_other"}],
            "reference_ambiguous_instance",
        ),
        ([{**row, "instance_id": "ins_missing"}], "reference_foreign_key"),
        ([{**row, "optimality_proven": False}], "unproven_optimal_value"),
        ([{**row, "solution_json": "[0,0,1]"}], "solver_objective_mismatch"),
        ([{**row, "solution_json": "[1,0,1]"}], "solver_solution_infeasible"),
    )
    for rows, expected in cases:
        issues = []
        validate_references(
            pa.Table.from_pylist(rows, schema=get_schema("reference_solutions")),
            _instance_table(identifier),
            issues,
        )
        assert expected in {issue.code for issue in issues}
