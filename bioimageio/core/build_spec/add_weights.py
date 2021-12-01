import os
from pathlib import Path
from shutil import copyfile
from typing import Dict, Optional, Union

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
    **weight_kwargs,
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
        tensorflow_version: the tensorflow version used for training the model.
            Only requred for models with tensorflow or keras weight format.
        opset_version: the opset version used in this model.
            Only requred for models with onnx weight format.
        weight_kwargs: additional keyword arguments for the weight.
    """
    model = load_raw_resource_description(model)

    # copy the weight path to the input model's root, otherwise it will
    # not be found when packaging the new model
    weight_out = os.path.join(model.root_path, Path(weight_uri).name)
    if Path(weight_out) != Path(weight_uri):
        copyfile(weight_uri, weight_out)

    new_weights, tmp_arch = _get_weights(
        weight_out,
        weight_type,
        root=Path("."),
        architecture=architecture,
        model_kwargs=model_kwargs,
        tensorflow_version=tensorflow_version,
        opset_version=opset_version,
        **weight_kwargs,
    )
    model.weights.update(new_weights)

    try:
        model_package = export_resource_package(model, output_path=output_path)
        model = load_raw_resource_description(model_package)
    except Exception as e:
        raise e
    finally:
        if tmp_arch is not None:
            os.remove(tmp_arch)
    return model
