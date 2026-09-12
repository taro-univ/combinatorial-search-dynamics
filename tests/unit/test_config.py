from pathlib import Path

import pytest

from llm_search_dynamics.config import CONFIG_GROUPS, compose_config, resolved_config


def test_default_hydra_composition_succeeds() -> None:
    assert compose_config().experiment.name == "pilot"


def test_resolved_config_contains_required_groups() -> None:
    config = resolved_config()
    assert set(CONFIG_GROUPS) <= config.keys()
    assert config["storage"]["implemented"] is True
    assert config["storage"]["parquet_dir"] == "data/raw/pilot"
    assert config["storage"]["zarr_store_path"] == "data/raw/pilot/observations.zarr"
    assert config["task"]["implemented"] is True
    assert config["tracking"]["implemented"] is True


def test_missing_config_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Hydra config file does not exist"):
        compose_config(config_dir=tmp_path)
