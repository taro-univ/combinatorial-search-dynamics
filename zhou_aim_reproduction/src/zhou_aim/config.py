from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def profile_path(project_root: str | Path, profile: str) -> Path:
    path = Path(project_root) / "configs" / f"{profile}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown profile {profile!r}: {path}")
    return path
