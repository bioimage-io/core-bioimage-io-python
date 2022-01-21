import enum
import json
import os
import sys
from glob import glob

from pathlib import Path
from pprint import pprint
from typing import List, Optional

import typer

from bioimageio.core import __version__, prediction, commands, resource_tests, load_raw_resource_description
from bioimageio.core.prediction_pipeline import get_weight_formats
from bioimageio.spec.__main__ import app, help_version as help_version_spec
from bioimageio.spec.model.raw_nodes import WeightsFormat

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore

try:
    from bioimageio.core.weight_converter import torch as torch_converter
except ImportError:
    torch_converter = None

try:
    from bioimageio.core.weight_converter import keras as keras_converter
except ImportError:
    keras_converter = None


# extend help/version string by core version
help_version_core = f"bioimageio.core {__version__}"
help_version = f"{help_version_spec}\n{help_version_core}"
# prevent rewrapping with \b\n: https://click.palletsprojects.com/en/7.x/documentation/#preventing-rewrapping
app.info.help = "\b\n" + help_version


@app.callback()
def callback():
    typer.echo(help_version)


@app.command()
def package(
    rdf_source: str = typer.Argument(..., help="RDF source as relative file path or URI"),
    path: Path = typer.Argument(Path() / "{src_name}-package.zip", help="Save package as"),
    weights_priority_order: Optional[List[str]] = typer.Option(
        None,
        "--weights-priority-order",
        "-wpo",
        help="For model packages only. "
        "If given only the first weights matching the given weight formats are included. "
        "Defaults to include all weights present in source.",
        show_default=False,
    ),
    verbose: bool = typer.Option(False, help="show traceback of exceptions"),
):
    # typer bug: typer returns empty tuple instead of None if weights_order_priority is not given
    weights_priority_order = weights_priority_order or None

    ret_code = commands.package(
        rdf_source=rdf_source, path=path, weights_priority_order=weights_priority_order, verbose=verbose
    )
    sys.exit(ret_code)


package.__doc__ = commands.package.__doc__


# if we want to use something like "choice" for the weight formats, we need to use an enum, see:
# https://github.com/tiangolo/typer/issues/182
WeightFormatEnum = enum.Enum("WeightFormatEnum", {wf: wf for wf in get_args(WeightsFormat)})
# Enum with in values does not work with click.Choice: https://github.com/pallets/click/issues/784
# so a simple Enum with auto int values is not an option:
# WeightFormatEnum = enum.Enum("WeightFormatEnum", get_args(WeightsFormat))


@app.command()
def test_model(
    model_rdf: str = typer.Argument(
        ..., help="Path or URL to the model resource description file (rdf.yaml) or zipped model."
    ),
    weight_format: Optional[WeightFormatEnum] = typer.Option(None, help="The weight format to use."),
    devices: Optional[List[str]] = typer.Option(None, help="Devices for running the model."),
    decimal: int = typer.Option(4, help="The test precision."),
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    summary = resource_tests.test_model(
        model_rdf,
        weight_format=None if weight_format is None else weight_format.value,
        devices=devices,
        decimal=decimal,
    )

    if weight_format is None:
        weight_formats = get_weight_formats()
        model_weight_formats = list(load_raw_resource_description(model_rdf).weights.keys())
        for wf in weight_formats:
            if wf in model_weight_formats:
                weight_format = wf
                break
        weight_format = "unknown" if weight_format is None else weight_format

    if summary["error"] is None:
        print(f"Model test for {model_rdf} using {weight_format} weight format has passed.")
        ret_code = 0
    else:
        print(f"Model test for {model_rdf} using {weight_format} weight format has FAILED!")
        pprint(summary)
        ret_code = 1
    sys.exit(ret_code)


test_model.__doc__ = resource_tests.test_model.__doc__


@app.command()
def test_resource(
    rdf: str = typer.Argument(
        ..., help="Path or URL to the resource description file (rdf.yaml) or zipped resource package."
    ),
    weight_format: Optional[WeightFormatEnum] = typer.Option(None, help="(for model only) The weight format to use."),
    devices: Optional[List[str]] = typer.Option(None, help="(for model only) Devices for running the model."),
    decimal: int = typer.Option(4, help="(for model only) The test precision."),
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    summary = resource_tests.test_resource(
        rdf, weight_format=None if weight_format is None else weight_format.value, devices=devices, decimal=decimal
    )
    if summary["error"] is None:
        print(f"Resource test for {rdf} has passed.")
        ret_code = 0
    else:
        print(f"Resource test for {rdf} has FAILED!")
        pprint(summary)
        ret_code = 1
    sys.exit(ret_code)


test_resource.__doc__ = resource_tests.test_resource.__doc__


@app.command()
def predict_image(
    model_rdf: Path = typer.Argument(
        ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
    ),
    inputs: List[Path] = typer.Option(..., help="Path(s) to the model input(s)."),
    outputs: List[Path] = typer.Option(..., help="Path(s) for saveing the model output(s)."),
    # NOTE: typer currently doesn't support union types, so we only support boolean here
    # padding: Optional[Union[str, bool]] = typer.Argument(
    #     None, help="Padding to apply in each dimension passed as json encoded string."
    # ),
    # tiling: Optional[Union[str, bool]] = typer.Argument(
    #     None, help="Padding to apply in each dimension passed as json encoded string."
    # ),
    padding: Optional[bool] = typer.Option(None, help="Whether to pad the image to a size suited for the model."),
    tiling: Optional[bool] = typer.Option(None, help="Whether to run prediction in tiling mode."),
    weight_format: Optional[WeightFormatEnum] = typer.Option(None, help="The weight format to use."),
    devices: Optional[List[str]] = typer.Option(None, help="Devices for running the model."),
):

    if isinstance(padding, str):
        padding = json.loads(padding.replace("'", '"'))
        assert isinstance(padding, dict)
    if isinstance(tiling, str):
        tiling = json.loads(tiling.replace("'", '"'))
        assert isinstance(tiling, dict)

    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    prediction.predict_image(
        model_rdf, inputs, outputs, padding, tiling, None if weight_format is None else weight_format.value, devices
    )


predict_image.__doc__ = prediction.predict_image.__doc__


@app.command()
def predict_images(
    model_rdf: Path = typer.Argument(
        ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
    ),
    input_pattern: str = typer.Argument(..., help="Glob pattern for the input images."),
    output_folder: str = typer.Argument(..., help="Folder to save the outputs."),
    output_extension: Optional[str] = typer.Argument(None, help="Optional output extension."),
    # NOTE: typer currently doesn't support union types, so we only support boolean here
    # padding: Optional[Union[str, bool]] = typer.Argument(
    #     None, help="Padding to apply in each dimension passed as json encoded string."
    # ),
    # tiling: Optional[Union[str, bool]] = typer.Argument(
    #     None, help="Padding to apply in each dimension passed as json encoded string."
    # ),
    padding: Optional[bool] = typer.Option(None, help="Whether to pad the image to a size suited for the model."),
    tiling: Optional[bool] = typer.Option(None, help="Whether to run prediction in tiling mode."),
    weight_format: Optional[WeightFormatEnum] = typer.Option(None, help="The weight format to use."),
    devices: Optional[List[str]] = typer.Option(None, help="Devices for running the model."),
):
    input_files = glob(input_pattern)
    input_names = [os.path.split(infile)[1] for infile in input_files]
    output_files = [os.path.join(output_folder, fname) for fname in input_names]
    if output_extension is not None:
        output_files = [f"{os.path.splitext(outfile)[0]}{output_extension}" for outfile in output_files]

    if isinstance(padding, str):
        padding = json.loads(padding.replace("'", '"'))
        assert isinstance(padding, dict)
    if isinstance(tiling, str):
        tiling = json.loads(tiling.replace("'", '"'))
        assert isinstance(tiling, dict)

    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    prediction.predict_images(
        model_rdf,
        input_files,
        output_files,
        padding=padding,
        tiling=tiling,
        weight_format=None if weight_format is None else weight_format.value,
        devices=devices,
        verbose=True,
    )


predict_images.__doc__ = prediction.predict_images.__doc__


if torch_converter is not None:

    @app.command()
    def convert_torch_weights_to_onnx(
        model_rdf: Path = typer.Argument(
            ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
        ),
        output_path: Path = typer.Argument(..., help="Where to save the onnx weights."),
        opset_version: Optional[int] = typer.Argument(12, help="Onnx opset version."),
        use_tracing: bool = typer.Option(True, help="Whether to use torch.jit tracing or scripting."),
        verbose: bool = typer.Option(True, help="Verbosity"),
    ):
        ret_code = torch_converter.convert_weights_to_onnx(model_rdf, output_path, opset_version, use_tracing, verbose)
        sys.exit(ret_code)

    convert_torch_weights_to_onnx.__doc__ = torch_converter.convert_weights_to_onnx.__doc__

    @app.command()
    def convert_torch_weights_to_torchscript(
        model_rdf: Path = typer.Argument(
            ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
        ),
        output_path: Path = typer.Argument(..., help="Where to save the torchscript weights."),
        use_tracing: bool = typer.Option(True, help="Whether to use torch.jit tracing or scripting."),
    ):
        ret_code = torch_converter.convert_weights_to_torchscript(model_rdf, output_path, use_tracing)
        sys.exit(ret_code)

    convert_torch_weights_to_torchscript.__doc__ = torch_converter.convert_weights_to_torchscript.__doc__


if keras_converter is not None:

    @app.command()
    def convert_keras_weights_to_tensorflow(
        model_rdf: Path = typer.Argument(
            ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
        ),
        output_path: Path = typer.Argument(..., help="Where to save the tensorflow weights."),
    ):
        ret_code = keras_converter.convert_weights_to_tensorflow_saved_model_bundle(model_rdf, output_path)
        sys.exit(ret_code)

    convert_keras_weights_to_tensorflow.__doc__ = (
        keras_converter.convert_weights_to_tensorflow_saved_model_bundle.__doc__
    )


if __name__ == "__main__":
    app()
