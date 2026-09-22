"""Causal features derived only from the current external task state."""

from __future__ import annotations

from typing import Any


def extract_external_features(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract current Hamming distance and bits without using future outcomes."""
    features: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        try:
            state = checkpoint["state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse checkpoint state for {checkpoint.get('checkpoint_id', '<unknown>')}"
            ) from exc
        common = {
            "instance_id": checkpoint["instance_id"],
            "trial_id": checkpoint["trial_id"],
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_index": int(checkpoint["checkpoint_index"]),
        }
        if "bits" in state:
            bits = [int(bit) for bit in state["bits"]]
            target = [int(bit) for bit in state["target_bits"]]
            if len(bits) != len(target) or not bits:
                raise ValueError("Checkpoint state bits and target must have equal non-zero length")
            if any(bit not in (0, 1) for bit in bits + target):
                raise ValueError("Checkpoint state must contain only binary values")
            features.append(
                {
                    **common,
                    "bits": bits,
                    "hamming_distance": sum(left != right for left, right in zip(bits, target)),
                }
            )
        elif "selected" in state:
            selected = [int(bit) for bit in state["selected"]]
            if not selected or any(bit not in (0, 1) for bit in selected):
                raise ValueError("Knapsack selected vector must be binary and non-empty")
            features.append(
                {
                    **common,
                    "selected": selected,
                    "total_weight": int(state["total_weight"]),
                    "total_value": int(state["total_value"]),
                    "optimality_gap": checkpoint.get("optimality_gap"),
                }
            )
        else:
            raise ValueError("Checkpoint state is not a supported external task state")
    return features
