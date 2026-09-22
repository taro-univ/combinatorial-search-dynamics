from importlib.metadata import version

import combinatorial_search_dynamics


def test_package_version_is_available() -> None:
    assert combinatorial_search_dynamics.__version__ == version("combinatorial-search-dynamics")
