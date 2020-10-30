from dataclasses import asdict

import typing
from pathlib import Path

from marshmallow import Schema, ValidationError, post_load, validate, validates, validates_schema
from pprint import pformat, pprint
from pybio.spec import fields, nodes
from pybio.spec.exceptions import PyBioValidationException
from pybio.spec.nodes import MagicShapeValue, MagicTensorsValue


class PyBioSchema(Schema):
    @post_load
    def make_object(self, data, **kwargs):
        if not data:
            return None

        this_type = getattr(nodes, self.__class__.__name__)
        try:
            return this_type(**data)
        except TypeError as e:
            e.args += (f"when initializing {this_type} from {self}",)
            raise e


class CiteEntry(PyBioSchema):
    text = fields.Str(required=True)
    doi = fields.Str(missing=None)
    url = fields.Str(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class BaseSpec(PyBioSchema):
    format_version = fields.Str(required=True)
    name = fields.Str(required=True)
    description = fields.Str(required=True)

    authors = fields.List(fields.Str)
    cite = fields.Nested(CiteEntry, many=True, required=True)

    git_repo = fields.Str(validate=validate.URL(schemes=["http", "https"]))
    tags = fields.List(fields.Str, required=True)
    license = fields.Str(required=True)

    documentation = fields.URI(required=True)
    covers = fields.List(fields.URI, missing=list)
    attachments = fields.Dict(fields.Str, missing=dict)

    config = fields.Dict(fields.Str, missing=dict)


class SpecWithKwargs(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Kwargs(fields.Str, missing=nodes.Kwargs)


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
    reference_input = fields.Str(missing=None)
    scale = fields.List(fields.Float, required=True)
    offset = fields.List(fields.Integer, required=True)

    @validates_schema
    def matching_lengths(self, data, **kwargs):
        scale = data["scale"]
        offset = data["offset"]
        if len(scale) != len(offset):
            raise PyBioValidationException(f"scale {scale} has to have same length as offset {offset}!")


class Array(PyBioSchema):
    name = fields.Str(required=True, validate=validate.Predicate("isidentifier"))
    description = fields.Str(required=True)
    axes = fields.Axes(missing=None)
    data_type = fields.Str(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Nested

    @validates_schema
    def axes_and_shape(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]
        if not isinstance(shape, MagicShapeValue) and shape and axes is None:
            raise PyBioValidationException("Axes may not be 'null', when shape is specified")


class InputArray(Array):
    shape = fields.Shape(InputShape, valid_magic_values=[MagicShapeValue.any], required=True)
    normalization = fields.Str(validate=validate.OneOf(["zero_mean_unit_variance"]), missing=None)

    @validates_schema
    def zero_batch_step(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]
        if not isinstance(shape, MagicShapeValue):
            if axes is None:
                raise PyBioValidationException("Axes field required when shape is specified")

            assert isinstance(axes, str), type(axes)

            bidx = axes.find("b")
            if isinstance(shape, tuple):
                # exact shape (with batch size 1)
                if bidx != -1 and shape[bidx] != 1:
                    raise PyBioValidationException("Input shape has to be one in the batch dimension.")

            elif isinstance(shape, nodes.InputShape):
                step = shape.step
                if bidx != -1 and shape.min[bidx] != 1:
                    raise PyBioValidationException("Input shape has to be one in the batch dimension.")

                if bidx != -1 and step[bidx] != 0:
                    raise PyBioValidationException(
                        "Input shape step has to be zero in the batch dimension (the batch dimension can always be "
                        "increased, but `step` should specify how to increase the minimal shape to find the largest "
                        "single batch shape)"
                    )
            else:
                raise PyBioValidationException(f"Unknown shape type {type(shape)}")


class OutputArray(Array):
    shape = fields.Shape(OutputShape, valid_magic_values=[MagicShapeValue.dynamic], required=True)
    halo = fields.List(fields.Integer, missing=None)

    @validates_schema
    def matching_halo_length(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            return
        elif isinstance(shape, tuple) or isinstance(shape, nodes.OutputShape):
            if len(halo) != len(shape):
                raise PyBioValidationException(f"halo {halo} has to have same length as shape {shape}!")
        elif not isinstance(shape, MagicShapeValue):
            raise NotImplementedError


class WithFileSource(PyBioSchema):
    source = fields.URI(required=True)
    sha256 = fields.Str(validate=validate.Length(equal=64))


class File(WithFileSource):
    pass


class Weight(WithFileSource):
    id = fields.Str(required=True, validate=validate.Predicate("isidentifier"))
    name = fields.Str(required=True)
    description = fields.Str(required=True)
    authors = fields.List(fields.Str)
    covers = fields.List(fields.URI, missing=list)
    test_inputs = fields.List(fields.URI(required=True), required=True)
    test_outputs = fields.List(fields.URI(required=True), required=True)
    timestamp = fields.DateTime(required=True)
    documentation = fields.URI(missing=None)
    tags = fields.List(fields.Str, required=True)
    attachments = fields.Dict(missing=dict)


class ModelSpec(BaseSpec):
    language = fields.Str(validate=validate.OneOf(["python", "java"]))
    framework = fields.Str(validate=validate.OneOf(["scikit-learn", "pytorch", "tensorflow"]))
    weights_format = fields.Str(validate=validate.OneOf(["pickle", "pytorch", "keras"]), required=True)
    dependencies = fields.Dependencies(missing=None)

    source = fields.ImportableSource(missing=None)
    sha256 = fields.Str(validate=validate.Length(equal=64), missing=None)
    kwargs = fields.Kwargs(fields.Str, missing=nodes.Kwargs)

    weights = fields.List(fields.Nested(Weight), required=True)
    inputs = fields.Tensors(InputArray, valid_magic_values=[MagicTensorsValue.any], many=True)
    outputs = fields.Tensors(
        OutputArray, valid_magic_values=[MagicTensorsValue.same, MagicTensorsValue.dynamic], many=True
    )
    config = fields.Dict(missing=dict)

    @validates("outputs")
    def validate_reference_input_names(self, data, **kwargs):
        pass  # todo validate_reference_input_names

    @validates_schema
    def language_and_framework_and_weights_format_match(self, data, **kwargs):
        names = ["language", "framework", "weights_format"]
        valid_combinations = {
            ("python", "scikit-learn", "pickle"): {"requires_source": False},
            ("python", "pytorch", "pytorch"): {"requires_source": True},
            ("python", "tensorflow", "keras"): {"requires_source": False},
            ("java", "tensorflow", "keras"): {"requires_source": False},
        }
        combination = tuple(data[name] for name in names)
        if combination not in valid_combinations:
            raise PyBioValidationException(f"invalid combination of {dict(zip(names, combination))}")

        if valid_combinations[combination]["requires_source"] and (data["source"] is None or data["sha256"] is None):
            raise PyBioValidationException(
                f"{dict(zip(names, combination))} require source code (and its sha256 hash) to be specified."
            )


class Model(SpecWithKwargs):
    spec = fields.SpecURI(ModelSpec, required=True)


class BioImageIoManifestModelEntry(Schema):
    id = fields.Str(required=True)
    source = fields.Str(validate=validate.URL(schemes=["http", "https"]))
    links = fields.List(fields.Str, missing=list)
    download_url = fields.Str(validate=validate.URL(schemes=["http", "https"]))


class BioImageIoManifest(Schema):
    format_version = fields.Str()
    config = fields.Dict()

    application = fields.List(fields.Dict)

    model = fields.List(fields.Nested(BioImageIoManifestModelEntry))


if __name__ == "__main__":
    from pybio.spec import load_model_config

    try:
        model = load_model_config(
            (Path(__file__) / "../../../specs/models/sklearnbased/RandomForestClassifier.model.yaml").as_uri()
        )
    except PyBioValidationException as e:
        pprint(e.normalized_messages(), width=280)
    else:
        pprint(asdict(model), width=280)
