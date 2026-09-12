import math

import pytest

from llm_search_dynamics.identifiers import artifact_id, checkpoint_id, instance_id, trial_id


def test_instance_id_ignores_dict_key_order() -> None:
    left = {"name": "探索", "values": [1, 2.5], "nested": {"a": True, "b": None}}
    right = {"nested": {"b": None, "a": True}, "values": [1, 2.5], "name": "探索"}
    assert instance_id(left) == instance_id(right)


def test_different_inputs_have_different_instance_ids() -> None:
    assert instance_id({"value": 1}) != instance_id({"value": 2})


def test_trial_id_is_deterministic() -> None:
    conditions = {"llm": "placeholder", "temperature": 0.0}
    assert trial_id("ins_example", conditions, 12) == trial_id("ins_example", conditions, 12)


def test_checkpoint_id_depends_on_token_position() -> None:
    assert checkpoint_id("trl_example", 1) != checkpoint_id("trl_example", 2)


def test_identifier_types_have_distinct_prefixes() -> None:
    assert instance_id({"value": 1}).startswith("ins_")
    assert trial_id("ins_example", {}, 1).startswith("trl_")
    assert checkpoint_id("trl_example", 1).startswith("chk_")
    assert artifact_id({"name": "observations"}).startswith("art_")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        instance_id({"value": value})


def test_non_json_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        instance_id({"value": object()})
