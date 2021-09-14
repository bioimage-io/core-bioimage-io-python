import os

import pytest

import bioimageio.spec as spec
from marshmallow import missing

from bioimageio.core.resource_io.io_ import load_raw_resource_description


def _test_build_spec(path, weight_type, tensorflow_version=None):
    from bioimageio.core.build_spec import build_model

    model_spec = load_raw_resource_description(path)
    assert isinstance(model_spec, spec.model.raw_nodes.Model)
    weight_source = model_spec.weights[weight_type].source

    cite = {entry.text: entry.doi if entry.url is missing else entry.url for entry in model_spec.cite}

    if weight_type == "pytorch_state_dict":
        source_path = os.path.join(model_spec.source.source_file.path)
        class_name = model_spec.source.callable_name
        model_source = f"{source_path}:{class_name}"
        weight_type_ = None  # the weight type can be auto-detected
    elif weight_type == "pytorch_script":
        model_source = None
        weight_type_ = "pytorch_script"  # the weight type CANNOT be auto-detcted
    else:
        model_source = None
        weight_type_ = None  # the weight type can be auto-detected

    dep_file = None if model_spec.dependencies is missing else model_spec.dependencies.file.path
    authors = [{"name": auth.name, "affiliation": auth.affiliation} for auth in model_spec.authors]
    covers = [cover.path for cover in model_spec.covers]
    kwargs = dict(
        source=model_source,
        model_kwargs=model_spec.kwargs,
        weight_uri=weight_source,
        test_inputs=[inp.path for inp in model_spec.test_inputs],
        test_outputs=[outp.path for outp in model_spec.test_outputs],
        name=model_spec.name,
        description=model_spec.description,
        authors=authors,
        tags=model_spec.tags,
        license=model_spec.license,
        documentation=model_spec.documentation,
        covers=covers,
        dependencies=dep_file,
        cite=cite,
        root=model_spec.root_path,
        weight_type=weight_type_,
    )
    if tensorflow_version is not None:
        kwargs["tensorflow_version"] = tensorflow_version
    raw_model = build_model(**kwargs)
    spec.model.schema.Model().dump(raw_model)


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_build_spec_pytorch(any_torch_model):
    _test_build_spec(any_torch_model, "pytorch_state_dict")


@pytest.mark.skipif(pytest.skip_torch, reason="requires torch")
def test_build_spec_torchscript(any_torchscript_model):
    _test_build_spec(any_torchscript_model, "pytorch_script")


@pytest.mark.skipif(pytest.skip_onnx, reason="requires onnx")
def test_build_spec_onnx(any_onnx_model):
    _test_build_spec(any_onnx_model, "onnx")


@pytest.mark.skipif(pytest.skip_tensorflow or pytest.tf_major_version != 1, reason="requires tensorflow 1")
def test_build_spec_keras(any_keras_model):
    _test_build_spec(any_keras_model, "keras_hdf5", tensorflow_version="1.12")  # todo: keras for tf 2??


@pytest.mark.skipif(pytest.skip_tensorflow, reason="requires tensorflow")
def test_build_spec_tf(any_tensorflow_model):
    _test_build_spec(
        any_tensorflow_model, "tensorflow_saved_model_bundle", tensorflow_version="1.12"
    )  # check tf version


@pytest.mark.skipif(pytest.skip_tensorflow, reason="requires tensorflow")
def test_build_spec_tfjs(any_tensorflow_js_model):
    _test_build_spec(any_tensorflow_js_model, "tensorflow_js", tensorflow_version="1.12")
