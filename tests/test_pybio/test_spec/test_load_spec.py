import pytest

from pybio.spec import load_spec_and_kwargs


def test_load_non_existing_spec(cache_path):
    spec_path = "some/none/existing/path/to/spec.model.yaml"

    with pytest.raises(FileNotFoundError):
        load_spec_and_kwargs(spec_path, cache_path=cache_path)


def test_load_non_valid_spec_name(cache_path):
    spec_path = "some/none/existing/path/to/spec.not_valid.yaml"

    with pytest.raises(ValueError):
        load_spec_and_kwargs(spec_path, cache_path=cache_path)
