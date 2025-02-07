from typing import Optional

from loguru import logger
from pydantic import DirectoryPath

from bioimageio.core._resource_tests import test_model
from bioimageio.spec import load_model_description, save_bioimageio_package_as_folder
from bioimageio.spec.model.v0_5 import ModelDescr, WeightsFormat


def increase_available_weight_formats(
    model_descr: ModelDescr,
    *,
    output_path: DirectoryPath,
    source_format: Optional[WeightsFormat] = None,
    target_format: Optional[WeightsFormat] = None,
) -> Optional[ModelDescr]:
    """Convert model weights to other formats and add them to the model description

    Args:
        output_path: Path to save updated model package to.
        source_format: convert from a specific weights format.
                       Default: choose automatically from any available.
        target_format: convert to a specific weights format.
                       Default: attempt to convert to any missing format.
        devices: Devices that may be used during conversion.

    Returns:
        - An updated model description if any converted weights were added.
        - `None` if no conversion was possible.
    """
    if not isinstance(model_descr, ModelDescr):
        raise TypeError(type(model_descr))

    # save model to local folder
    output_path = save_bioimageio_package_as_folder(
        model_descr, output_path=output_path
    )
    # reload from local folder to make sure we do not edit the given model
    _model_descr = load_model_description(output_path)
    assert isinstance(_model_descr, ModelDescr)
    model_descr = _model_descr
    del _model_descr

    if source_format is None:
        available = set(model_descr.weights.available_formats)
    else:
        available = {source_format}

    if target_format is None:
        missing = set(model_descr.weights.missing_formats)
    else:
        missing = {target_format}

    originally_missing = set(missing)

    if "pytorch_state_dict" in available and "onnx" in missing:
        from .pytorch_to_onnx import convert

        try:
            model_descr.weights.onnx = convert(
                model_descr,
                output_path=output_path,
                use_tracing=False,
            )
        except Exception as e:
            logger.error(e)
        else:
            available.add("onnx")
            missing.discard("onnx")

    if "pytorch_state_dict" in available and "torchscript" in missing:
        from .pytorch_to_torchscript import convert

        try:
            model_descr.weights.torchscript = convert(
                model_descr,
                output_path=output_path,
                use_tracing=False,
            )
        except Exception as e:
            logger.error(e)
        else:
            available.add("torchscript")
            missing.discard("torchscript")

    if missing:
        logger.warning(
            f"Converting from any of the available weights formats {available} to any"
            + f" of {missing} is not yet implemented. Please create an issue at"
            + " https://github.com/bioimage-io/core-bioimage-io-python/issues/new/choose"
            + " if you would like bioimageio.core to support a particular conversion."
        )

    if originally_missing == missing:
        logger.warning("failed to add any converted weights")
        return None
    else:
        logger.info(f"added weights formats {originally_missing - missing}")
        test_model(model_descr).display()
        return model_descr
