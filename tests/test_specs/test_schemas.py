from marshmallow import ValidationError
from pytest import raises

from pybio.spec import schema


class TestPreprocessing:
    class TestZeroMeanUniVarianceKwargs:
        def test_invalid(self):
            with raises(ValidationError):
                schema.Preprocessing().load({"name": "zero_mean_unit_variance"})

        def test_mode_fixed(self):
            schema.Preprocessing().load(
                {"name": "zero_mean_unit_variance", "kwargs": {"mode": "fixed", "mean": 1, "std": 2, "axes": "xy"}}
            )
