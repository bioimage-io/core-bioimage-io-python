import enum
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path
from pprint import pformat
from typing import List, Optional, get_args

import typer
from typing_extensions import Annotated

from bioimageio.core import __version__, prediction, resource_tests
from bioimageio.spec import load_description, save_bioimageio_package
from bioimageio.spec.collection import CollectionDescr
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr
from bioimageio.spec.model.v0_5 import WeightsFormat
from bioimageio.spec.notebook import NotebookDescr
from bioimageio.spec.summary import ValidationSummary

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bioimageio.core.weight_converter import torch as torch_converter
except ImportError:
    torch_converter = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bioimageio.core.weight_converter import keras as keras_converter
except ImportError:
    keras_converter = None

help_version = f"""bioimageio.core {__version__}
bioimageio.spec {__version__}
implementing:
\tcollection RDF {CollectionDescr.implemented_format_version}
\tdataset RDF {DatasetDescr.implemented_format_version}
\tmodel RDF {ModelDescr.implemented_format_version}
\tnotebook RDF {NotebookDescr.implemented_format_version}"""


# prevent rewrapping with \b\n: https://click.palletsprojects.com/en/7.x/documentation/#preventing-rewrapping
app = typer.Typer(
    help="\b\n" + help_version,
    context_settings={"help_option_names": ["-h", "--help", "--version"]},  # make --version display help with version
)  # https://typer.tiangolo.com/


@app.callback()
def callback():
    typer.echo(help_version)


# if we want to use something like "choice" for the weight formats, we need to use an enum, see:
# https://github.com/tiangolo/typer/issues/182
WeightsFormatEnum = enum.Enum("WeightsFormatEnum", {wf: wf for wf in get_args(WeightsFormat)})
# Enum with in values does not work with click.Choice: https://github.com/pallets/click/issues/784
# so a simple Enum with auto int values is not an option:
# WeightsFormatEnum = enum.Enum("WeightsFormatEnum", get_args(WeightsFormat))


@app.command()
def package(
    rdf_source: Annotated[str, typer.Argument(help="RDF source as relative file path or URI")],
    path: Annotated[Path, typer.Argument(help="Save package as")] = Path() / "bioimageio-package.zip",
    weights_priority_order: Annotated[
        Optional[List[WeightsFormatEnum]],
        typer.Option(
            "--weights-priority-order",
            "-wpo",
            help="For model packages only. "
            "If given only the first weights matching the given weight formats are included. "
            "Defaults to include all weights present in source.",
            show_default=False,
        ),
    ] = None,
    # verbose: Annotated[bool, typer.Option(help="show traceback of exceptions")] = False,
):
    # typer bug: typer returns empty tuple instead of None if weights_order_priority is not given
    weights_priority_order = weights_priority_order or None  # TODO: check if this is still the case

    _ = save_bioimageio_package(rdf_source, output_path=path, weights_priority_order=weights_priority_order)


def _log_test_summaries(summaries: List[ValidationSummary], msg: str):
    # todo: improve logging of multiple test summaries
    ret_code = 0
    for summary in summaries:
        print(f"{summary['name']}: {summary['status']}")
        if summary["status"] != "passed":
            s = {
                k: v
                for k, v in summary.items()
                if k not in ("name", "status", "bioimageio_spec_version", "bioimageio_core_version")
            }
            tb = s.pop("traceback")
            if tb:
                print("traceback:")
                print("".join(tb))

            def show_part(part, show):
                if show:
                    line = f"{part}: "
                    print(line + pformat(show, width=min(80, 120 - len(line))).replace("\n", " " * len(line) + "\n"))

            for part in ["error", "warnings", "source_name"]:
                show_part(part, s.pop(part, None))

            for part in sorted(s.keys()):
                show_part(part, s[part])

            ret_code = 1

    if ret_code:
        result = "FAILED!"
        icon = "❌"
    else:
        result = "passed."
        icon = "✔️"

    print(msg.format(icon=icon, result=result))
    return ret_code


@app.command()
def test_model(
    model_rdf: Annotated[
        str, typer.Argument(help="Path or URL to the model resource description file (rdf.yaml) or zipped model.")
    ],
    weight_format: Annotated[Optional[WeightsFormatEnum], typer.Option(help="The weight format to use.")] = None,
    devices: Annotated[Optional[List[str]], typer.Option(help="Devices for running the model.")] = None,
    decimal: Annotated[int, typer.Option(help="The test precision.")] = 4,
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    devices = devices or None

    summaries = resource_tests.test_model(
        model_rdf,
        weight_format=None if weight_format is None else weight_format.value,
        devices=devices,
        decimal=decimal,
    )
    print(f"\ntesting model {model_rdf}...")
    ret_code = _log_test_summaries(summaries, f"\n{{icon}} Model {model_rdf} {{result}}")
    sys.exit(ret_code)


test_model.__doc__ = resource_tests.test_model.__doc__


@app.command()
def test_resource(
    rdf: str = typer.Argument(
        ..., help="Path or URL to the resource description file (rdf.yaml) or zipped resource package."
    ),
    weight_format: Optional[WeightsFormatEnum] = typer.Option(None, help="(for model only) The weight format to use."),
    devices: Optional[List[str]] = typer.Option(None, help="(for model only) Devices for running the model."),
    decimal: int = typer.Option(4, help="(for model only) The test precision."),
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    summaries = resource_tests.test_description(
        rdf, weight_format=None if weight_format is None else weight_format.value, devices=devices, decimal=decimal
    )
    print(f"\ntesting {rdf}...")
    ret_code = _log_test_summaries(summaries, f"{{icon}} Resource test for {rdf} has {{result}}")
    sys.exit(ret_code)


test_resource.__doc__ = resource_tests.test_description.__doc__


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
    weight_format: Optional[WeightsFormatEnum] = typer.Option(None, help="The weight format to use."),
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
    weight_format: Optional[WeightsFormatEnum] = typer.Option(None, help="The weight format to use."),
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
        torch_converter.convert_weights_to_torchscript(model_rdf, output_path, use_tracing)
        sys.exit(0)

    convert_torch_weights_to_torchscript.__doc__ = torch_converter.convert_weights_to_torchscript.__doc__


if keras_converter is not None:

    @app.command()
    def convert_keras_weights_to_tensorflow(
        model_rdf: Annotated[
            Path, typer.Argument(help="Path to the model resource description file (rdf.yaml) or zipped model.")
        ],
        output_path: Annotated[Path, typer.Argument(help="Where to save the tensorflow weights.")],
    ):
        rd = load_description(model_rdf)
        ret_code = keras_converter.convert_weights_to_tensorflow_saved_model_bundle(rd, output_path)
        sys.exit(ret_code)

    convert_keras_weights_to_tensorflow.__doc__ = (
        keras_converter.convert_weights_to_tensorflow_saved_model_bundle.__doc__
    )


if __name__ == "__main__":
    app()
