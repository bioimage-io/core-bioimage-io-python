import numpy
import pytest
from marshmallow import Schema, ValidationError
from numpy.testing import assert_equal
from pytest import raises

from pybio.spec import fields


class SchemaWithIntegerArray(Schema):
    array = fields.Array(fields.Integer(strict=True))


@pytest.fixture
def schema_with_integer_array():
    return SchemaWithIntegerArray()


class TestArray:
    def test_unequal_nesting_depth(self, schema_with_integer_array):
        with raises(ValidationError):
            schema_with_integer_array.load({"array": [[1, 2], 3]})

    def test_uneuqal_sublen(self, schema_with_integer_array):
        with raises(ValidationError):
            schema_with_integer_array.load({"array": [[1, 2], [3]]})

    def test_scalar(self, schema_with_integer_array):
        data = 1
        expected = {"array": data}
        actual = schema_with_integer_array.load({"array": data})
        assert_equal(actual, expected)

    def test_invalid_scalar(self, schema_with_integer_array):
        data = "invalid"
        with raises(ValidationError):
            schema_with_integer_array.load({"array": data})

    def test_2d(self, schema_with_integer_array):
        data = [[1, 2], [3, 4]]
        expected = {"array": numpy.array(data, dtype=int)}
        actual = schema_with_integer_array.load({"array": data})
        assert_equal(actual, expected)

    def test_wrong_dtype(self, schema_with_integer_array):
        data = [[1, 2], [3, 4.5]]
        with raises(ValidationError):
            schema_with_integer_array.load({"array": data})
