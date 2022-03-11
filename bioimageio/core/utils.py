from functools import wraps
from typing import Literal, Set, Type


def skip_on(exception: Type[Exception], reason: str):
    """adapted from https://stackoverflow.com/a/63522579"""
    import pytest

    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(reason)

        return wrapper

    return decorator_func


TensorName = str
FixedMode = Literal["fixed"]
SampleMode = Literal["per_sample"]
DatasetMode = Literal["per_dataset"]
Mode = Literal[FixedMode, SampleMode, DatasetMode]

FIXED: FixedMode = "fixed"
PER_SAMPLE: SampleMode = "per_sample"
PER_DATASET: DatasetMode = "per_dataset"
MODES: Set[Mode] = {FIXED, PER_SAMPLE, PER_DATASET}
