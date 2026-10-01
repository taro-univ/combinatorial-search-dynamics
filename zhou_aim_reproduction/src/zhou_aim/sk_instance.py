from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SKInstance:
    couplings: np.ndarray
    seed: int
    j_scale: float = 1.0

    @property
    def n(self) -> int:
        return int(self.couplings.shape[0])

    def validation(self) -> dict[str, float | int | bool]:
        upper = self.couplings[np.triu_indices(self.n, k=1)]
        expected_variance = self.j_scale**2 / self.n
        return {
            "n": self.n,
            "seed": self.seed,
            "symmetric": bool(np.array_equal(self.couplings, self.couplings.T)),
            "diagonal_zero": bool(np.all(np.diag(self.couplings) == 0.0)),
            "sample_mean": float(upper.mean()),
            "sample_variance_ddof1": float(upper.var(ddof=1)),
            "expected_variance": float(expected_variance),
            "variance_ratio": float(upper.var(ddof=1) / expected_variance),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({"seed": self.seed, "j_scale": self.j_scale}, sort_keys=True)
        np.savez_compressed(path, couplings=self.couplings, metadata=np.asarray(metadata))

    @classmethod
    def load(cls, path: str | Path) -> SKInstance:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            couplings = np.asarray(payload["couplings"], dtype=np.float64)
        instance = cls(couplings=couplings, **metadata)
        report = instance.validation()
        if not report["symmetric"] or not report["diagonal_zero"]:
            raise ValueError(f"Invalid SK instance in {path}: {report}")
        return instance


def generate_sk(n: int, seed: int, j_scale: float = 1.0) -> SKInstance:
    if n < 2:
        raise ValueError("SK size n must be at least 2")
    if j_scale <= 0:
        raise ValueError("j_scale must be positive")
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n, n), dtype=np.float64)
    indices = np.triu_indices(n, k=1)
    matrix[indices] = rng.normal(0.0, j_scale / np.sqrt(n), size=len(indices[0]))
    matrix[(indices[1], indices[0])] = matrix[indices]
    return SKInstance(couplings=matrix, seed=seed, j_scale=j_scale)


def instance_metadata(instance: SKInstance) -> dict[str, object]:
    return {
        **asdict(instance),
        "couplings": "stored separately",
        "validation": instance.validation(),
    }
