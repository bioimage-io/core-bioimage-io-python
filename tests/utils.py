"""utils to test bioimageio.core"""

import os
from functools import wraps
from typing import Any, Protocol, Sequence, Type

import pytest


class ParameterSet(Protocol):
    def __init__(self, values: Sequence[Any], marks: Any, id: str) -> None:
        super().__init__()


class test_func(Protocol):
    def __call__(*args: Any, **kwargs: Any): ...


def skip_on(exception: Type[Exception], reason: str):
    """adapted from https://stackoverflow.com/a/63522579"""
    import pytest

    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f: test_func):
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(reason)

        return wrapper

    return decorator_func


expensive_test = pytest.mark.skipif(
    (run := os.getenv("RUN_EXPENSIVE_TESTS")) != "true",
    reason="Skipping expensive test (enable by RUN_EXPENSIVE_TESTS='true')",
)
