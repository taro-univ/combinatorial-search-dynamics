from importlib.metadata import version

import llm_search_dynamics


def test_package_version_is_available() -> None:
    assert llm_search_dynamics.__version__ == version("llm-search-dynamics")
