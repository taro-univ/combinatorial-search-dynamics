from combinatorial_search_dynamics.config import repository_root, validate_repository


def test_repository_validation_succeeds() -> None:
    assert validate_repository(repository_root()) == []
