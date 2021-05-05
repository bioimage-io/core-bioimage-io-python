import numpy
from datetime import datetime, timezone
from marshmallow import ValidationError
from numpy.testing import assert_equal
from pytest import raises

from pybio.spec import fields, raw_nodes, schema


class TestArray:
    def test_unequal_nesting_depth(self):
        with raises(ValidationError):
            fields.Array(fields.Integer(strict=True)).deserialize([[1, 2], 3])

    def test_uneuqal_sublen(self):
        with raises(ValidationError):
            fields.Array(fields.Integer(strict=True)).deserialize([[1, 2], [3]])

    def test_scalar(self):
        data = 1
        expected = data
        actual = fields.Array(fields.Integer(strict=True)).deserialize(data)
        assert_equal(actual, expected)

    def test_invalid_scalar(self):
        data = "invalid"
        with raises(ValidationError):
            fields.Array(fields.Integer(strict=True)).deserialize(data)

    def test_2d(self):
        data = [[1, 2], [3, 4]]
        expected = numpy.array(data, dtype=int)
        actual = fields.Array(fields.Integer(strict=True)).deserialize(data)
        assert_equal(actual, expected)

    def test_wrong_dtype(self):
        data = [[1, 2], [3, 4.5]]
        with raises(ValidationError):
            fields.Array(fields.Integer(strict=True)).deserialize(data)


class TestDateTime:
    def test_datetime_from_str(self):
        timestamp = "2019-12-11T12:22:32+00:00"
        expected = datetime.fromisoformat(timestamp)
        actual = fields.DateTime().deserialize(timestamp)
        assert expected == actual

    def test_datetime_from_datetime(self):
        expected = datetime.now()
        assert expected == fields.DateTime().deserialize(expected)

    def test_datetime_iso_with_zulu_offset(self):
        timestamp_non_zulu = "2019-12-11T12:22:32+00:00"
        timestamp_zulu = "2019-12-11T12:22:32Z"
        expected = datetime(2019, 12, 11, 12, 22, 32, tzinfo=timezone.utc)
        actual1 = fields.DateTime().deserialize(timestamp_non_zulu)
        actual2 = fields.DateTime().deserialize(timestamp_zulu)
        assert expected == actual1
        assert expected == actual2


class TestShape:
    def test_explicit_input_shape(self):
        data = [1, 2, 3]
        expected = data
        actual = fields.InputShape().deserialize(data)
        assert expected == actual

    def test_explicit_output_shape(self):
        data = [1, 2, 3]
        expected = data
        actual = schema.OutputTensor().fields["shape"].deserialize(data)
        assert expected == actual

    def test_min_step_input_shape(self):
        data = {"min": [1, 2, 3], "step": [0, 1, 3]}
        expected = raw_nodes.ImplicitInputShape(**data)
        actual = fields.InputShape().deserialize(data)
        assert expected == actual

    def test_todo_output_shape(self):
        # todo: output shape with schema (implicit shape)
        pass

    def test_explicit_input_shape_schema(self):
        class MySchema(schema.Schema):
            shape = fields.InputShape()

        data = {
            "shape": [1, 2, 3],
        }
        expected = data
        actual = MySchema().load(data)
        assert expected == actual
