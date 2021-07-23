import os
from pathlib import Path

from bioimageio.spec.model import schema
from bioimageio.spec.model.converters import maybe_convert
from bioimageio.spec.shared import yaml
from bioimageio.spec.shared.utils import resolve_uri


def _test_unet(url, weight_type, weight_source):
    from bioimageio.core.build_spec import build_spec

    config_path = Path(resolve_uri(url))
    assert os.path.exists(config_path), config_path
    model_spec = yaml.load(Path(config_path))
    model_spec = maybe_convert(model_spec)

    if weight_source is None:
        weight_source = model_spec["weights"][weight_type]["source"]
    test_inputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_input.npy"
    ]
    test_outputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_output.npy"
    ]

    cite = {entry["text"]: entry["doi"] if "doi" in entry else entry["url"] for entry in model_spec["cite"]}

    if weight_type == "pytorch_state_dict":
        # we need to download the source code for pytorch weights
        # and then pass the model source as local path and model class
        source_url = ("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_specs/models/"
                      "unet2d_nuclei_broad/unet2d.py")
        resolve_uri(source_url)
        model_source = "unet2d.py:UNet2d"
        weight_type_ = None   # the weight type can be auto-detected
    elif weight_type == "pytorch_script":
        model_source = None
        weight_type_ = "pytorch_script"  # the weight type CANNOT be auto-detcted
    else:
        model_source = None
        weight_type_ = None  # the weight type can be auto-detected

    raw_model = build_spec(
        source=model_source,
        model_kwargs=model_spec["kwargs"],
        weight_uri=weight_source,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        name=model_spec["name"],
        description=model_spec["description"],
        authors=model_spec["authors"],
        tags=model_spec["tags"],
        license=model_spec["license"],
        documentation=model_spec["documentation"],
        covers=model_spec["covers"],
        dependencies=model_spec["dependencies"],
        cite=cite,
        root=config_path.parent,
        weight_type=weight_type_
    )
    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(model_spec)


def test_build_spec_pytorch(unet2d_nuclei_broad_model_url):
    _test_unet(unet2d_nuclei_broad_model_url, "pytorch_state_dict", None)


def test_build_spec_onnx(unet2d_nuclei_broad_model_url):
    weight_source = ("https://github.com/bioimage-io/spec-bioimage-io/blob/main/example_specs/models/"
                     "unet2d_nuclei_broad/weights.onnx")
    _test_unet(unet2d_nuclei_broad_model_url, "onnx", weight_source)


def test_build_spec_torchscript(unet2d_nuclei_broad_model_url):
    weight_source = ("https://github.com/bioimage-io/spec-bioimage-io/blob/main/example_specs/"
                     "models/unet2d_nuclei_broad/weights.pt")
    _test_unet(unet2d_nuclei_broad_model_url, "pytorch_script", weight_source)


def _test_frunet(url, weight_source):
    from bioimageio.core.build_spec import build_spec

    config_path = resolve_uri(url)
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    test_inputs = ["https://github.com/deepimagej/models/raw/master/fru-net_sev_segmentation/exampleImage.npy"]
    test_outputs = ["https://github.com/deepimagej/models/raw/master/fru-net_sev_segmentation/resultImage.npy"]
    cite = {entry["text"]: entry["doi"] if "doi" in entry else entry["url"] for entry in source["cite"]}

    raw_model = build_spec(
        weight_uri=weight_source,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        name=source["name"],
        description=source["description"],
        authors=source["authors"],
        tags=source["tags"],
        license=source["license"],
        documentation=source["documentation"],
        covers=source["covers"],
        cite=cite,
        tensorflow_version="1.12",
    )

    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(source)


def test_build_spec_keras(FruNet_model_url):
    weight_source = "https://zenodo.org/record/4156050/files/fully_residual_dropout_segmentation.h5"
    _test_frunet(FruNet_model_url, weight_source)


def test_build_spec_tf(FruNet_model_url):
    weight_source = "https://zenodo.org/record/4156050/files/tensorflow_saved_model_bundle.zip"
    _test_frunet(FruNet_model_url, weight_source)


def test_build_spec_tfjs(FruNet_model_url):
    weight_source = (
        "https://raw.githubusercontent.com/deepimagej/tensorflow-js-models/main/"
        "fru-net_sev_segmentation_tf_js_model/model.json"
    )
    _test_frunet(FruNet_model_url, weight_source)
