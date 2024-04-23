# TODO: move to tests/
from functools import wraps
from typing import Any, Protocol, Type


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
