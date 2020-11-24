from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Mapping, Union, get_type_hints

from marshmallow import Schema, ValidationError, post_load, validate, validates, validates_schema

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
    text = fields.String(required=True)
    doi = fields.String(missing=None)
    url = fields.String(missing=None)

    @validates_schema
    def doi_or_url(self, data, **kwargs):
        if data["doi"] is None and data["url"] is None:
            raise ValidationError("doi or url needs to be specified in a citation")


class Spec(PyBioSchema):
    format_version = fields.String(
        validate=validate.OneOf(get_type_hints(nodes.Spec)["format_version"].__args__), required=True
    )
    name = fields.String(required=True)
    description = fields.String(required=True)

    authors = fields.List(fields.String)
    cite = fields.Nested(CiteEntry, many=True, required=True)

    git_repo = fields.String(validate=validate.URL(schemes=["http", "https"]))
    tags = fields.List(fields.String, required=True)
    license = fields.String(required=True)

    documentation = fields.URI(required=True)
    covers = fields.List(fields.URI, missing=list)
    attachments = fields.Dict(fields.String, missing=dict)

    config = fields.Dict(missing=dict)


class SpecWithKwargs(PyBioSchema):
    spec: fields.SpecURI
    kwargs = fields.Kwargs(fields.String, missing=nodes.Kwargs)


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


class Tensor(PyBioSchema):
    name = fields.String(required=True, validate=validate.Predicate("isidentifier"))
    description = fields.String(required=True)
    axes = fields.Axes(required=True)  # todo check if null is ok (it shouldn't)
    data_type = fields.String(required=True)
    data_range = fields.Tuple((fields.Float(allow_nan=True), fields.Float(allow_nan=True)))

    shape: fields.Shape


class Preprocessing(PyBioSchema):
    name = fields.String(validate=validate.OneOf(get_type_hints(nodes.Preprocessing)["name"].__args__), required=True)
    kwargs = fields.Kwargs(fields.String, missing=nodes.Kwargs)

    @post_load
    def make_object(self, data, **kwargs):
        if not data:
            return None

        # camel_case_name = data["name"].title()
        this_type = getattr(nodes, self.__class__.__name__)
        try:
            return this_type(**data)
        except TypeError as e:
            e.args += (f"when initializing {this_type} from {self}",)
            raise e

    class ZeroMeanUniVarianceKwargs(Schema):  # not pybio schema, only returning a validated dict, no specific node
        mode = fields.String(validate=validate.OneOf(("fixed", "per_dataset", "per_sample")), required=True)
        axes = fields.Axes(valid_axes="czyx")  # todo: check for input if these axes are a subset
        mean = fields.Array(fields.Float(), missing=None)  # todo: check if means match input axes (for mode 'fixed')
        std = fields.Array(fields.Float(), missing=None)

        @validates_schema
        def mean_and_std_match_mode(self, data, **kwargs):
            if data["mode"] == "fixed" and (data["mean"] is None or data["std"] is None):
                raise PyBioValidationException(
                    "`kwargs` for 'zero_mean_unit_variance' preprocessing with `mode` 'fixed' require additional `kwargs`: `mean` and `std`."
                )
            elif data["mode"] != "fixed" and (data["mean"] is not None or data["std"] is not None):
                raise PyBioValidationException(
                    "`kwargs`: `mean` and `std` for 'zero_mean_unit_variance' preprocessing are only valid for `mode` 'fixed'."
                )

    @validates_schema
    def kwargs_match_selected_preprocessing_name(self, data, **kwargs):
        if data["name"] == "zero_mean_unit_variance":
            kwargs_validation_errors = self.ZeroMeanUniVarianceKwargs().validate(data["kwargs"])
        else:
            raise NotImplementedError(
                "Validating the 'name' field should not allow you to get here, unless you just added a new "
                "preprocessing 'name'. So what kwargs go with it?"
            )

        if kwargs_validation_errors:
            raise PyBioValidationException(f"Invalid `kwargs` for '{data['name']}': {kwargs_validation_errors}")


class InputTensor(Tensor):
    shape = fields.Shape(InputShape, required=True)
    preprocessing = fields.List(fields.Nested(Preprocessing), missing=list)

    @validates_schema
    def zero_batch_step_and_one_batch_size(self, data, **kwargs):
        axes = data["axes"]
        shape = data["shape"]

        bidx = axes.find("b")
        if bidx == -1:
            return

        if isinstance(shape, nodes.InputShape):
            step = shape.step
            shape = shape.min

        elif isinstance(shape, tuple):
            step = (0,) * len(shape)
        else:
            raise PyBioValidationException(f"Unknown shape type {type(shape)}")

        if step[bidx] != 0:
            raise PyBioValidationException(
                "Input shape step has to be zero in the batch dimension (the batch dimension can always be "
                "increased, but `step` should specify how to increase the minimal shape to find the largest "
                "single batch shape)"
            )

        if shape[bidx] != 1:
            raise PyBioValidationException("Input shape has to be 1 in the batch dimension b.")


class OutputTensor(Tensor):
    shape = fields.Shape(OutputShape, required=True)
    halo = fields.Halo()

    # halo = fields.Method(deserialize="load_halo")
    #
    # def load_halo(self, value):
    #     _halo = fields.List(fields.Integer, missing=None)
    #     if value is None:
    #         return (0,) * len(self.shape)
    #
    #     return value

    @validates_schema
    def matching_halo_length(self, data, **kwargs):
        shape = data["shape"]
        halo = data["halo"]
        if halo is None:
            return
        elif isinstance(shape, tuple) or isinstance(shape, nodes.OutputShape):
            if len(halo) != len(shape):
                raise PyBioValidationException(f"halo {halo} has to have same length as shape {shape}!")
        else:
            raise NotImplementedError(type(shape))


class WithFileSource(PyBioSchema):
    source = fields.URI(required=True)
    sha256 = fields.String(validate=validate.Length(equal=64))


class Weight(WithFileSource):
    id = fields.String(required=True, validate=validate.Predicate("isidentifier"))
    name = fields.String(required=True)
    description = fields.String(required=True)
    authors = fields.List(fields.String)
    covers = fields.List(fields.URI, missing=list)
    test_inputs = fields.List(fields.URI(required=True), required=True)
    test_outputs = fields.List(fields.URI(required=True), required=True)
    documentation = fields.URI(missing=None)
    tags = fields.List(fields.String, required=True)
    attachments = fields.Dict(missing=dict)


class Model(Spec):
    language = fields.String(validate=validate.OneOf(get_type_hints(nodes.Model)["language"].__args__), required=True)
    framework = fields.String(validate=validate.OneOf(get_type_hints(nodes.Model)["framework"].__args__), required=True)
    dependencies = fields.Dependencies(missing=None)
    timestamp = fields.DateTime(required=True)

    source = fields.ImportableSource(missing=None)
    sha256 = fields.String(validate=validate.Length(equal=64), missing=None)
    kwargs = fields.Kwargs(fields.String, missing=nodes.Kwargs)

    weights = fields.Dict(
        fields.String(
            validate=validate.OneOf(get_type_hints(nodes.Model)["weights"].__args__[0].__args__), required=True
        ),
        fields.Nested(Weight),
        required=True,
    )
    inputs = fields.Tensors(InputTensor, valid_magic_values=[MagicTensorsValue.any], many=True)
    outputs = fields.Tensors(
        OutputTensor, valid_magic_values=[MagicTensorsValue.same, MagicTensorsValue.dynamic], many=True
    )
    config = fields.Dict(missing=dict)

    @validates("outputs")
    def validate_reference_input_names(self, data, **kwargs):
        pass  # todo validate_reference_input_names

    @validates_schema
    def language_and_framework_match(self, data, **kwargs):
        field_names = ("language", "framework")
        valid_combinations = {
            ("python", "scikit-learn"): {"requires_source": False},
            ("python", "pytorch"): {"requires_source": True},
            ("python", "tensorflow"): {"requires_source": False},
            ("java", "tensorflow"): {"requires_source": False},
        }
        combination = tuple(data[name] for name in field_names)
        if combination not in valid_combinations:
            raise PyBioValidationException(f"invalid combination of {dict(zip(field_names, combination))}")

        if valid_combinations[combination]["requires_source"] and (data["source"] is None or data["sha256"] is None):
            raise PyBioValidationException(
                f"{dict(zip(field_names, combination))} require source code (and its sha256 hash) to be specified."
            )

    @validates_schema
    def source_specified_if_required(self, data, **kwargs):
        if data["source"] is not None:
            return

        weight_format_requires_source = {"pickle": True, "pytorch": True, "keras": False}
        require_source = {wf for wf in data["weights"] if weight_format_requires_source[wf]}
        if require_source:
            raise PyBioValidationException(
                f"These specified weight formats require source code to be specified: {require_source}"
            )


class BioImageIoManifestModelEntry(Schema):
    id = fields.String(required=True)
    source = fields.String(validate=validate.URL(schemes=["http", "https"]))
    links = fields.List(fields.String, missing=list)
    download_url = fields.String(validate=validate.URL(schemes=["http", "https"]))


class BioImageIoManifest(Schema):
    format_version = fields.String(
        validate=validate.OneOf(get_type_hints(nodes.Spec)["format_version"].__args__), required=True
    )
    config = fields.Dict()

    application = fields.List(fields.Dict)

    model = fields.List(fields.Nested(BioImageIoManifestModelEntry))


# helper schemas
class File(WithFileSource):
    pass


class Resource(PyBioSchema):
    uri = fields.URI(required=False)


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
