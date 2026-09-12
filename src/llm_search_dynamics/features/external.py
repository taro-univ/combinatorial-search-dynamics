"""Causal features derived only from the current external task state."""

from __future__ import annotations

from typing import Any


def extract_external_features(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract current Hamming distance and bits without using future outcomes."""
    features: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        try:
            state = checkpoint["state"]
            bits = [int(bit) for bit in state["bits"]]
            target = [int(bit) for bit in state["target_bits"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse checkpoint state for {checkpoint.get('checkpoint_id', '<unknown>')}"
            ) from exc
        if len(bits) != len(target) or not bits:
            raise ValueError("Checkpoint state bits and target must have equal non-zero length")
        if any(bit not in (0, 1) for bit in bits + target):
            raise ValueError("Checkpoint state must contain only binary values")
        features.append(
            {
                "instance_id": checkpoint["instance_id"],
                "trial_id": checkpoint["trial_id"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_index": int(checkpoint["checkpoint_index"]),
                "bits": bits,
                "hamming_distance": sum(left != right for left, right in zip(bits, target)),
            }
        )
    return features
