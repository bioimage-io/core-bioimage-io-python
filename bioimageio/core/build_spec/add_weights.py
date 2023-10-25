import os
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Union

from pydantic import DirectoryPath, FilePath

from bioimageio.core import export_resource_package
from bioimageio.core.io import FileSource, download, read_description, write_package_as_folder
from bioimageio.spec.model import AnyModel, v0_5

from .build_model import _get_weights


def add_weights(
    model: Union[AnyModel, FileSource],
    weight_file: FileSource,
    output_path: DirectoryPath,
    *,
    weight_type: Optional[v0_5.WeightsFormat] = None,
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
        weight_file: the weight file to be added
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
    downloaded_weight_file = download(weight_file)
    output_path = write_package_as_folder(model, output_path=output_path)

    # copy the weight path to the input model's root, otherwise it will
    # not be found when packaging the new model
    weight_out: FilePath = output_path / downloaded_weight_file.original_file_name  # noqa: F821
    _ = copyfile(downloaded_weight_file.path, weight_out)

    new_weights, tmp_arch = _get_weights(
        weight_out,
        weight_type,
        root=output_path,
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
        model = read_description(model_package)
    except Exception as e:
        raise e
    finally:
        # clean up tmp files
        if Path(weight_out).absolute() != Path(weight_file).absolute():
            os.remove(weight_out)
        if tmp_arch is not None:
            os.remove(tmp_arch)
        # for some reason the weights are also copied to the cwd.
        # not sure why this happens, but it needs to be cleaned up, unless these are the input weigths
        weights_cwd = Path(os.path.split(weight_file)[1])
        if weights_cwd.exists() and weights_cwd.absolute() != Path(weight_file).absolute():
            os.remove(weights_cwd)
    return model
