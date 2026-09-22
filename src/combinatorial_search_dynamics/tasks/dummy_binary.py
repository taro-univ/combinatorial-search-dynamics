"""Deterministic binary-target task used only to validate the Phase 2 pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BinaryInstance:
    bit_count: int
    initial_bits: tuple[int, ...]
    target_bits: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bit_count": self.bit_count,
            "initial_bits": list(self.initial_bits),
            "target_bits": list(self.target_bits),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BinaryInstance:
        return cls(
            bit_count=int(value["bit_count"]),
            initial_bits=tuple(int(bit) for bit in value["initial_bits"]),
            target_bits=tuple(int(bit) for bit in value["target_bits"]),
        )


@dataclass(frozen=True)
class BinaryState:
    bits: tuple[int, ...]
    target_bits: tuple[int, ...]
    step: int
    max_steps: int


class DummyBinaryTask:
    """Flip one bit per action to minimize Hamming distance to a target."""

    name = "dummy_binary"
    version = "1"
    objective_sense = "minimize"

    def __init__(self, *, bit_count: int, max_steps: int) -> None:
        if bit_count <= 0:
            raise ValueError("bit_count must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.bit_count = bit_count
        self.max_steps = max_steps

    def generate_instance(self, seed: int, **params: Any) -> BinaryInstance:
        del params
        rng = random.Random(seed)
        initial = tuple(rng.randrange(2) for _ in range(self.bit_count))
        target = tuple(rng.randrange(2) for _ in range(self.bit_count))
        if initial == target:
            target = (1 - target[0], *target[1:])
        return BinaryInstance(self.bit_count, initial, target)

    def initial_state(self, instance: BinaryInstance) -> BinaryState:
        self._validate_instance(instance)
        return BinaryState(instance.initial_bits, instance.target_bits, 0, self.max_steps)

    def legal_actions(self, state: BinaryState) -> tuple[int, ...]:
        self._validate_state(state)
        return tuple(range(self.bit_count))

    def apply_action(self, state: BinaryState, action: int) -> BinaryState:
        self._validate_state(state)
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or action not in range(self.bit_count)
        ):
            raise ValueError(f"action must be an integer in [0, {self.bit_count - 1}]")
        bits = list(state.bits)
        bits[action] = 1 - bits[action]
        return BinaryState(tuple(bits), state.target_bits, state.step + 1, state.max_steps)

    def is_feasible(self, state: BinaryState) -> bool:
        try:
            self._validate_state(state)
        except (TypeError, ValueError):
            return False
        return True

    def is_terminal(self, state: BinaryState, reference_value: float | None = None) -> bool:
        return self.is_success(state, reference_value) or state.step >= state.max_steps

    def objective(self, state: BinaryState) -> float:
        self._validate_state(state)
        return float(
            sum(left != right for left, right in zip(state.bits, state.target_bits, strict=True))
        )

    def is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate < incumbent - 1e-9

    def objective_gap(self, value: float, reference: float, *, epsilon: float = 1e-9) -> float:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return max(0.0, value - reference) / max(abs(reference), epsilon)

    def is_success(self, state: BinaryState, reference_value: float | None = None) -> bool:
        del reference_value
        return abs(self.objective(state)) <= 1e-9

    def serialize_state(self, state: BinaryState) -> dict[str, Any]:
        self._validate_state(state)
        return {
            "bits": list(state.bits),
            "target_bits": list(state.target_bits),
            "step": state.step,
            "max_steps": state.max_steps,
        }

    def deserialize_state(self, value: dict[str, Any]) -> BinaryState:
        state = BinaryState(
            bits=tuple(int(bit) for bit in value["bits"]),
            target_bits=tuple(int(bit) for bit in value["target_bits"]),
            step=int(value["step"]),
            max_steps=int(value["max_steps"]),
        )
        self._validate_state(state)
        return state

    def parse_action(self, text: str) -> int:
        try:
            action = int(text.strip())
        except ValueError as exc:
            raise ValueError("action text must contain an integer bit position") from exc
        if action not in range(self.bit_count):
            raise ValueError(f"action must be in [0, {self.bit_count - 1}]")
        return action

    def _validate_instance(self, instance: BinaryInstance) -> None:
        if instance.bit_count != self.bit_count:
            raise ValueError("instance bit_count does not match task configuration")
        self._validate_bits(instance.initial_bits)
        self._validate_bits(instance.target_bits)

    def _validate_state(self, state: BinaryState) -> None:
        if not isinstance(state, BinaryState):
            raise TypeError("state must be BinaryState")
        self._validate_bits(state.bits)
        self._validate_bits(state.target_bits)
        if state.step < 0 or state.max_steps <= 0:
            raise ValueError("state step budget is invalid")

    def _validate_bits(self, bits: tuple[int, ...]) -> None:
        if len(bits) != self.bit_count or any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"bits must contain exactly {self.bit_count} binary values")
