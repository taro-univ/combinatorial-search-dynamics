"""Small registry for implemented task adapters."""

from __future__ import annotations

from llm_search_dynamics.tasks.base import TaskProtocol
from llm_search_dynamics.tasks.dummy_binary import DummyBinaryTask


def get_task(name: str, *, bit_count: int, max_steps: int) -> TaskProtocol:
    if name == DummyBinaryTask.name:
        return DummyBinaryTask(bit_count=bit_count, max_steps=max_steps)
    raise KeyError(f"Unknown task: {name}")
