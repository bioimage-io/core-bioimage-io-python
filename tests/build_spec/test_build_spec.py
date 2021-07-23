import os
from pathlib import Path

from bioimageio.spec.model import schema
from bioimageio.spec.model.converters import maybe_convert
from bioimageio.spec.shared import yaml


def test_build_spec_pytorch(unet2d_nuclei_broad_latest_path):
    from bioimageio.spec.build_spec import build_spec

    config_path = unet2d_nuclei_broad_latest_path
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = source["weights"]["pytorch_state_dict"]["source"]
    test_inputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_input.npy"
    ]
    test_outputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_output.npy"
    ]

    cite = {entry["text"]: entry["doi"] if "doi" in entry else entry["url"] for entry in source["cite"]}

    raw_model = build_spec(
        source=source["source"],
        model_kwargs=source["kwargs"],
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
        dependencies=source["dependencies"],
        cite=cite,
        root=config_path.parent,
    )
    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(source)


def test_build_spec_onnx(unet2d_nuclei_broad_latest_path):
    from bioimageio.spec.build_spec import build_spec

    config_path = unet2d_nuclei_broad_latest_path
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = (
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/"
        + "unet2d_nuclei_broad/weights.onnx"
    )
    test_inputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_input.npy"
    ]
    test_outputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_output.npy"
    ]

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
        dependencies=source["dependencies"],
        cite=cite,
    )
    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(source)


def test_build_spec_torchscript(unet2d_nuclei_broad_latest_path):
    from bioimageio.spec.build_spec import build_spec

    config_path = unet2d_nuclei_broad_latest_path
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = (
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/" + "unet2d_nuclei_broad/weights.pt"
    )
    test_inputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_input.npy"
    ]
    test_outputs = [
        "https://github.com/bioimage-io/pytorch-bioimage-io/raw/master/specs/models/unet2d_nuclei_broad/test_output.npy"
    ]

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
        dependencies=source["dependencies"],
        cite=cite,
        weight_type="pytorch_script",
    )
    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(source)


def test_build_spec_keras(FruNet_model_url):
    from bioimageio.spec.build_spec import _get_local_path, build_spec

    config_path = _get_local_path(FruNet_model_url)
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = "https://zenodo.org/record/4156050/files/fully_residual_dropout_segmentation.h5"
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


def test_build_spec_tf(FruNet_model_url):
    from bioimageio.spec.build_spec import _get_local_path, build_spec

    config_path = _get_local_path(FruNet_model_url)
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = "https://zenodo.org/record/4156050/files/tensorflow_saved_model_bundle.zip"
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


def test_build_spec_tfjs(FruNet_model_url):
    from bioimageio.spec.build_spec import _get_local_path, build_spec

    config_path = _get_local_path(FruNet_model_url)
    assert os.path.exists(config_path), config_path
    source = yaml.load(Path(config_path))
    source = maybe_convert(source)

    weight_source = (
        "https://raw.githubusercontent.com/deepimagej/tensorflow-js-models/main/"
        "fru-net_sev_segmentation_tf_js_model/model.json"
    )
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
    )

    serialized = schema.Model().dump(raw_model)
    assert type(serialized) == type(source)
