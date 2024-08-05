"""These functions implement the logic of the bioimageio command line interface
defined in the `cli` module."""

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

from typing_extensions import Literal

from bioimageio.core import test_description
from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    save_bioimageio_package,
    save_bioimageio_package_as_folder,
)
from bioimageio.spec.model.v0_5 import WeightsFormat

WeightFormatArg = Literal[WeightsFormat, "all"]


def test(
    descr: Union[ResourceDescr, InvalidDescr],
    *,
    weight_format: WeightFormatArg = "all",
    devices: Optional[Union[str, Sequence[str]]] = None,
    decimal: int = 4,
):
    """test a bioimageio resource

    Args:
        source: Path or URL to the bioimageio resource description file
                (bioimageio.yaml or rdf.yaml) or to a zipped resource
        weight_format: (model only) The weight format to use
        devices: Device(s) to use for testing
        decimal: Precision for numerical comparisons
    """
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        sys.exit(1)

    summary = test_description(
        descr,
        weight_format=None if weight_format == "all" else weight_format,
        devices=[devices] if isinstance(devices, str) else devices,
        decimal=decimal,
    )
    summary.display()
    sys.exit(0 if summary.status == "passed" else 1)


def validate_format(
    descr: Union[ResourceDescr, InvalidDescr],
):
    """validate the meta data format of a bioimageio resource

    Args:
        descr: a bioimageio resource description
    """
    descr.validation_summary.display()
    sys.exit(0 if descr.validation_summary.status == "passed" else 1)


def package(
    descr: ResourceDescr, path: Path, *, weight_format: WeightFormatArg = "all"
):
    """Save a resource's metadata with its associated files.

    Note: If `path` does not have a `.zip` suffix this command will save the
          package as an unzipped folder instead.

    Args:
        descr: a bioimageio resource description
        path: output path
        weight-format: include only this single weight-format (if not 'all').
    """
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        raise ValueError("resource description is invalid")

    if weight_format == "all":
        weights_priority_order = None
    else:
        weights_priority_order = (weight_format,)

    if path.suffix == ".zip":
        _ = save_bioimageio_package(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )
    else:
        _ = save_bioimageio_package_as_folder(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )


# TODO: add convert command(s)
# if torch_converter is not None:

#     @app.command()
#     def convert_torch_weights_to_onnx(
#         model_rdf: Path = typer.Argument(
#             ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
#         ),
#         output_path: Path = typer.Argument(..., help="Where to save the onnx weights."),
#         opset_version: Optional[int] = typer.Argument(12, help="Onnx opset version."),
#         use_tracing: bool = typer.Option(True, help="Whether to use torch.jit tracing or scripting."),
#         verbose: bool = typer.Option(True, help="Verbosity"),
#     ):
#         ret_code = torch_converter.convert_weights_to_onnx(model_rdf, output_path, opset_version, use_tracing, verbose)
#         sys.exit(ret_code)

#     convert_torch_weights_to_onnx.__doc__ = torch_converter.convert_weights_to_onnx.__doc__

#     @app.command()
#     def convert_torch_weights_to_torchscript(
#         model_rdf: Path = typer.Argument(
#             ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
#         ),
#         output_path: Path = typer.Argument(..., help="Where to save the torchscript weights."),
#         use_tracing: bool = typer.Option(True, help="Whether to use torch.jit tracing or scripting."),
#     ):
#         torch_converter.convert_weights_to_torchscript(model_rdf, output_path, use_tracing)
#         sys.exit(0)

#     convert_torch_weights_to_torchscript.__doc__ = torch_converter.convert_weights_to_torchscript.__doc__


# if keras_converter is not None:

#     @app.command()
#     def convert_keras_weights_to_tensorflow(
#         model_rdf: Annotated[
#             Path, typer.Argument(help="Path to the model resource description file (rdf.yaml) or zipped model.")
#         ],
#         output_path: Annotated[Path, typer.Argument(help="Where to save the tensorflow weights.")],
#     ):
#         rd = load_description(model_rdf)
#         ret_code = keras_converter.convert_weights_to_tensorflow_saved_model_bundle(rd, output_path)
#         sys.exit(ret_code)

#     convert_keras_weights_to_tensorflow.__doc__ = (
#         keras_converter.convert_weights_to_tensorflow_saved_model_bundle.__doc__
#     )
