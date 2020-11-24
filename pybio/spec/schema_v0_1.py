from marshmallow import Schema, ValidationError, validates, validates_schema

from pybio.spec import fields, nodes
from pybio.spec.exceptions import PyBioValidationException
from pybio.spec.nodes import MagicShapeValue, MagicTensorsValue


class CiteEntry(Schema):
    text = fields.String(required=True)
    doi = fields.String(missing=None)
    url = fields.String(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class BaseSpec(Schema):
    name = fields.String(required=True)
    format_version = fields.String(required=True)
    description = fields.String(required=True)
    cite = fields.Nested(CiteEntry, many=True, required=True)
    authors = fields.List(fields.String(required=True))
    documentation = fields.Path(required=True)
    tags = fields.List(fields.String, required=True)
    license = fields.String(required=True)

    language = fields.String(required=True)
    framework = fields.String(missing=None)
    source = fields.String(required=True)
    required_kwargs = fields.List(fields.String, missing=list)
    optional_kwargs = fields.Dict(fields.String, missing=dict)

    test_input = fields.Path(missing=None)
    test_output = fields.Path(missing=None)
    covers = fields.List(fields.Path, missing=list)


class SpecWithKwargs(Schema):
    spec: fields.SpecURI
    kwargs = fields.Dict(missing=dict)


class InputShape(Schema):
    min = fields.List(fields.Integer, required=True)
    step = fields.List(fields.Integer, required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        min_ = data["min"]
        step = data["step"]
        if min_ is None or step is None:
            return

        if len(min_) != len(step):
            raise PyBioValidationException(
                f"'min' and 'step' have to have the same length! (min: {min_}, step: {step})"
            )


class OutputShape(Schema):
    reference_input = fields.String(missing=None)
    scale = fields.List(fields.Float, required=True)
    offset = fields.List(fields.Integer, required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        scale = data["scale"]
        offset = data["offset"]
        if len(scale) != len(offset):
            raise PyBioValidationException(f"scale {scale} has to have same length as offset {offset}!")


class Array(Schema):
    name = fields.String(required=True)
    axes = fields.Axes(missing=None)
    data_type = fields.String(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Nested


class InputArray(Array):
    shape = fields.Shape(InputShape, valid_magic_values=[MagicShapeValue.any], required=True)


class OutputArray(Array):
    shape = fields.Shape(OutputShape, valid_magic_values=[MagicShapeValue.dynamic], required=True)
    halo = fields.List(fields.Integer, missing=None)


class TransformationSpec(BaseSpec):
    dependencies = fields.Dependencies(required=True)
    inputs = fields.Tensors(
        InputArray, valid_magic_values=[MagicTensorsValue.any, MagicTensorsValue.dynamic], required=True
    )
    outputs = fields.Tensors(
        OutputArray, valid_magic_values=[MagicTensorsValue.same, MagicTensorsValue.dynamic], required=True
    )


class Transformation(SpecWithKwargs):
    spec = fields.SpecURI(TransformationSpec, required=True)


class Weights(Schema):
    source = fields.URI(required=True)
    hash = fields.Dict()


class Prediction(Schema):
    weights = fields.Nested(Weights, missing=None)
    dependencies = fields.Dependencies(missing=None)
    preprocess = fields.Nested(Transformation, many=True, missing=list)
    postprocess = fields.Nested(Transformation, many=True, missing=list)


class ReaderSpec(BaseSpec):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Tensors(OutputArray, valid_magic_values=[MagicTensorsValue.dynamic], required=True)


class Reader(SpecWithKwargs):
    spec = fields.SpecURI(ReaderSpec)
    transformations = fields.List(fields.Nested(Transformation), missing=list)


class SamplerSpec(BaseSpec):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Tensors(OutputArray, valid_magic_values=[MagicTensorsValue.dynamic], missing=None)


class Sampler(SpecWithKwargs):
    spec = fields.SpecURI(SamplerSpec)
    readers = fields.List(fields.Nested(Reader, required=True), required=True)


class Optimizer(Schema):
    source = fields.String(required=True)
    required_kwargs = fields.List(fields.String, missing=list)
    optional_kwargs = fields.Dict(fields.String, missing=dict)


class Setup(Schema):
    samplers = fields.List(fields.Nested(Sampler, required=True), required=True)
    preprocess = fields.Nested(Transformation, many=True, missing=list)
    postprocess = fields.Nested(Transformation, many=True, missing=list)
    losses = fields.Nested(Transformation, many=True, missing=list)
    optimizer = fields.Nested(Optimizer, missing=None)


# class Training(Schema):
#     setup = fields.Nested(Setup)
#     source = fields.String(required=True)
#     required_kwargs = fields.List(fields.String, missing=list)
#     optional_kwargs = fields.Dict(fields.String, missing=dict)
#     dependencies = fields.Dependencies(required=True)
#     description = fields.String(missing=None)


class Model(BaseSpec):
    prediction = fields.Nested(Prediction)
    inputs = fields.Tensors(InputArray, valid_magic_values=[MagicTensorsValue.any], many=True)
    outputs = fields.Tensors(
        OutputArray, valid_magic_values=[MagicTensorsValue.same, MagicTensorsValue.dynamic], many=True
    )
    # training = fields.Nested(Training, missing=None)
    training = fields.Dict(missing=None)

    config = fields.Dict(missing=dict)

# class Model(SpecWithKwargs):
#     spec = fields.SpecURI(ModelSpec, required=True)


# helper schemas
class File(Schema):
    source = fields.URI(required=True)
    hash = fields.Dict()
