from marshmallow import Schema, ValidationError, validates_schema

from pybio.spec import fields
from pybio.spec.exceptions import PyBioValidationException


class PyBioSchema(Schema):
    bioimageio_description: str = ""


class CiteEntry(PyBioSchema):
    text = fields.String(required=True)
    doi = fields.String(missing=None)
    url = fields.String(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class BaseSpec(PyBioSchema):
    name = fields.String(required=True)
    format_version = fields.String(required=True)
    description = fields.String(required=True)
    cite = fields.Nested(CiteEntry(), many=True, required=True)
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


class SpecWithKwargs(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Dict(missing=dict)


class InputShape(PyBioSchema):
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


class OutputShape(PyBioSchema):
    reference_input = fields.String(missing=None)
    scale = fields.List(fields.Float, required=True)
    offset = fields.List(fields.Integer, required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        scale = data["scale"]
        offset = data["offset"]
        if len(scale) != len(offset):
            raise PyBioValidationException(f"scale {scale} has to have same length as offset {offset}!")


class Array(PyBioSchema):
    name = fields.String(required=True)
    axes = fields.Axes(missing=None)
    data_type = fields.String(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Nested


class InputArray(Array):
    shape = fields.Union([fields.ExplicitShape(), fields.Nested(InputShape)], required=True)


class OutputArray(Array):
    shape = fields.Union([fields.ExplicitShape(), fields.Nested(OutputShape)], required=True)
    halo = fields.List(fields.Integer, missing=None)


class TransformationSpec(BaseSpec):
    dependencies = fields.Dependencies(required=True)
    inputs = fields.Nested(InputArray, required=True)
    outputs = fields.Nested(OutputArray, required=True)


class Transformation(SpecWithKwargs):
    spec = fields.SpecURI(TransformationSpec, required=True)


class Weights(PyBioSchema):
    source = fields.URI(required=True)
    hash = fields.Dict()


class Prediction(PyBioSchema):
    weights = fields.Nested(Weights, missing=None)
    dependencies = fields.Dependencies(missing=None)
    preprocess = fields.Nested(Transformation, many=True, missing=list)
    postprocess = fields.Nested(Transformation, many=True, missing=list)


class ReaderSpec(BaseSpec):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Nested(OutputArray, required=True)


class Reader(SpecWithKwargs):
    spec = fields.SpecURI(ReaderSpec)
    transformations = fields.List(fields.Nested(Transformation), missing=list)


class SamplerSpec(BaseSpec):
    dependencies = fields.Dependencies(missing=None)
    outputs = fields.Nested(OutputArray, missing=None)


class Sampler(SpecWithKwargs):
    spec = fields.SpecURI(SamplerSpec)
    readers = fields.List(fields.Nested(Reader, required=True), required=True)


class Optimizer(PyBioSchema):
    source = fields.String(required=True)
    required_kwargs = fields.List(fields.String, missing=list)
    optional_kwargs = fields.Dict(fields.String, missing=dict)


class Setup(PyBioSchema):
    samplers = fields.List(fields.Nested(Sampler, required=True), required=True)
    preprocess = fields.Nested(Transformation, many=True, missing=list)
    postprocess = fields.Nested(Transformation, many=True, missing=list)
    losses = fields.Nested(Transformation, many=True, missing=list)
    optimizer = fields.Nested(Optimizer, missing=None)


class Model(BaseSpec):
    prediction = fields.Nested(Prediction)
    inputs = fields.Nested(InputArray, many=True)
    outputs = fields.Nested(OutputArray, many=True)
    training = fields.Dict(missing=None)

    config = fields.Dict(missing=dict)
