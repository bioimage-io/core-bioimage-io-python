import os

import bioimageio.spec as spec
from marshmallow import missing


def _test_build_spec(path, weight_type, tf_version=None):
    from bioimageio.core.build_spec import build_model

    model_spec, root_path = spec.ensure_raw_resource_description(path, None, update_to_current_format=False)
    weight_source = model_spec.weights[weight_type].source

    cite = {entry.text: entry.doi if entry.url is missing else entry.url for entry in model_spec.cite}

    if weight_type == "pytorch_state_dict":
        source_path = os.path.join(model_spec.source.source_file.path)
        class_name = model_spec.source.callable_name
        model_source = f"{source_path}:{class_name}"
        weight_type_ = None   # the weight type can be auto-detected
    elif weight_type == "pytorch_script":
        model_source = None
        weight_type_ = "pytorch_script"  # the weight type CANNOT be auto-detcted
    else:
        model_source = None
        weight_type_ = None  # the weight type can be auto-detected

    dep_file = model_spec.dependencies.file.path
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
        root=root_path,
        weight_type=weight_type_
    )
    if tf_version is not None:
        kwargs["tf_version"] = tf_version
    raw_model = build_model(**kwargs)
    spec.model.schema.Model().dump(raw_model)


def test_build_spec_pytorch(unet2d_nuclei_broad_model):
    _test_build_spec(unet2d_nuclei_broad_model, "pytorch_state_dict")


def test_build_spec_onnx(unet2d_nuclei_broad_model):
    _test_build_spec(unet2d_nuclei_broad_model, "onnx")


def test_build_spec_torchscript(unet2d_nuclei_broad_model):
    _test_build_spec(unet2d_nuclei_broad_model, "pytorch_script")


def test_build_spec_keras(FruNet_model):
    _test_build_spec(FruNet_model, "keras_hdf5", tensorflow_version="1.12")


def test_build_spec_tf(FruNet_model):
    _test_build_spec(FruNet_model, "tensorflow_saved_model_bundle", tensorflow_version="1.12")


def test_build_spec_tfjs(FruNet_model):
    _test_build_spec(FruNet_model, "tensorflow_js", tensorflow_version="1.12")
