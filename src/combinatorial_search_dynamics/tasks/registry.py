"""Small registry for implemented task adapters."""

from __future__ import annotations

from combinatorial_search_dynamics.tasks.base import TaskProtocol
from combinatorial_search_dynamics.tasks.dummy_binary import DummyBinaryTask
from combinatorial_search_dynamics.tasks.knapsack import KnapsackTask


def get_task(
    name: str, *, bit_count: int | None = None, max_steps: int, **params: object
) -> TaskProtocol:
    if name == DummyBinaryTask.name:
        if bit_count is None:
            raise ValueError("dummy_binary requires bit_count")
        return DummyBinaryTask(bit_count=bit_count, max_steps=max_steps)
    if name == KnapsackTask.name:
        return KnapsackTask(max_steps=max_steps, **params)
    raise KeyError(f"Unknown task: {name}")
