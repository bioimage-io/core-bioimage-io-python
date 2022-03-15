import os
from pathlib import Path
from shutil import copyfile
from typing import Dict, Optional, Union, List

from bioimageio.core import export_resource_package, load_raw_resource_description
from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription
from .build_model import _get_weights


def add_weights(
    model: Union[RawResourceDescription, os.PathLike, str],
    weight_uri: Union[str, Path],
    output_path: Union[str, Path],
    *,
    weight_type: Optional[str] = None,
    architecture: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
    tensorflow_version: Optional[str] = None,
    opset_version: Optional[str] = None,
    pytorch_version: Optional[str] = None,
    attachments: Optional[Dict[str, Union[str, List[str]]]] = None,
):
    """Add weight entry to bioimage.io model.

    Args:
        model: the resource description of the model to which the weight format is added
        weight_uri: the weight file to be added
        output_path: where to serialize the new model with additional weight format
        weight_type: the format of the weights to be added
        architecture: the file with the source code for the model architecture and the corresponding class.
            Only required for models with pytorch_state_dict weight format.
        model_kwargs: the keyword arguments for the model class.
            Only required for models with pytorch_state_dict weight format.
        tensorflow_version: the tensorflow version for this model. Only for tensorflow or keras weights.
        opset_version: the opset version for this model. Only for onnx weights.
        pytorch_version: the pytorch version for this model. Only for pytoch_state_dict or torchscript weights.
        attachments: extra weight specific attachments.
    """
    model = load_raw_resource_description(model)

    # copy the weight path to the input model's root, otherwise it will
    # not be found when packaging the new model
    weight_out = os.path.join(model.root_path, Path(weight_uri).name)
    if Path(weight_out).absolute() != Path(weight_uri).absolute():
        copyfile(weight_uri, weight_out)

    new_weights, tmp_arch = _get_weights(
        weight_out,
        weight_type,
        root=Path("."),
        architecture=architecture,
        model_kwargs=model_kwargs,
        tensorflow_version=tensorflow_version,
        opset_version=opset_version,
        pytorch_version=pytorch_version,
        attachments=attachments,
    )
    model.weights.update(new_weights)

    try:
        model_package = export_resource_package(model, output_path=output_path)
        model = load_raw_resource_description(model_package)
    except Exception as e:
        raise e
    finally:
        # clean up tmp files
        if Path(weight_out).absolute() != Path(weight_uri).absolute():
            os.remove(weight_out)
        if tmp_arch is not None:
            os.remove(tmp_arch)
        # for some reason the weights are also copied to the cwd.
        # not sure why this happens, but it needs to be cleaned up, unless these are the input weigths
        weights_cwd = Path(os.path.split(weight_uri)[1])
        if weights_cwd.exists() and weights_cwd.absolute() != Path(weight_uri).absolute():
            os.remove(weights_cwd)
    return model
