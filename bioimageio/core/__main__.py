import enum
import sys
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from bioimageio.core import __version__
from bioimageio.core import test_description as _test_description
from bioimageio.core import test_model as _test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.collection import CollectionDescr
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr
from bioimageio.spec.notebook import NotebookDescr

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
    context_settings={
        "help_option_names": ["-h", "--help", "--version"]
    },  # make --version display help with version
)  # https://typer.tiangolo.com/


@app.callback()
def callback():
    typer.echo(help_version)


# if we want to use something like "choice" for the weight formats, we need to use an enum, see:
# https://github.com/tiangolo/typer/issues/182


class WeightsFormatEnum(enum.Enum):
    keras_hdf5 = "keras_hdf5"
    onnx = "onnx"
    pytorch_state_dict = "pytorch_state_dict"
    tensorflow_js = "tensorflow_js"
    tensorflow_saved_model_bundle = "tensorflow_saved_model_bundle"
    torchscript = "torchscript"


# Enum with int values does not work with click.Choice: https://github.com/pallets/click/issues/784
# so a simple Enum with auto int values is not an option.


@app.command()
def package(
    source: Annotated[str, typer.Argument(help="path or url to a bioimageio RDF")],
    path: Annotated[Path, typer.Argument(help="Save package as")] = Path(
        "bioimageio-package.zip"
    ),
    weights_priority_order: Annotated[
        Optional[List[WeightsFormatEnum]],
        typer.Option(
            "--weights-priority-order",
            "-wpo",
            help="For model packages only. "
            + "If given, only the first matching weights entry is included. "
            + "Defaults to including all weights present in source.",
            show_default=False,
        ),
    ] = None,
):
    # typer bug: typer returns empty tuple instead of None if weights_order_priority is not given
    weights_priority_order = (
        weights_priority_order or None
    )  # TODO: check if this is still the case

    _ = save_bioimageio_package(
        source,
        output_path=path,
        weights_priority_order=(
            None
            if weights_priority_order is None
            else [wpo.name for wpo in weights_priority_order]
        ),
    )


@app.command()
def test_model(
    model_rdf: Annotated[
        str,
        typer.Argument(
            help="Path or URL to the model resource description file (rdf.yaml) or zipped model."
        ),
    ],
    weight_format: Annotated[
        Optional[WeightsFormatEnum], typer.Option(help="The weight format to use.")
    ] = None,
    devices: Annotated[
        Optional[List[str]], typer.Option(help="Devices for running the model.")
    ] = None,
    decimal: Annotated[int, typer.Option(help="The test precision.")] = 4,
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    devices = devices or None

    summary = _test_model(
        model_rdf,
        weight_format=None if weight_format is None else weight_format.value,
        devices=devices,
        decimal=decimal,
    )
    print(f"\ntesting model {model_rdf}...")
    print(summary.format())
    sys.exit(0 if summary.status == "passed" else 1)


test_model.__doc__ = _test_model.__doc__


@app.command()
def test_resource(
    rdf: Annotated[
        str,
        typer.Argument(
            help="Path or URL to the resource description file (rdf.yaml) or zipped resource package."
        ),
    ],
    weight_format: Annotated[
        Optional[WeightsFormatEnum],
        typer.Option(help="(for model only) The weight format to use."),
    ] = None,
    devices: Annotated[
        Optional[List[str]],
        typer.Option(help="(for model only) Devices for running the model."),
    ] = None,
    decimal: Annotated[
        int, typer.Option(help="(for model only) The test precision.")
    ] = 4,
):
    # this is a weird typer bug: default devices are empty tuple although they should be None
    if devices is None or len(devices) == 0:
        devices = None

    summary = _test_description(
        rdf,
        weight_format=None if weight_format is None else weight_format.value,
        devices=devices,
        decimal=decimal,
    )
    print(summary.format())
    sys.exit(0 if summary.status == "passed" else 1)


test_resource.__doc__ = _test_description.__doc__


# TODO: add predict commands
# @app.command()
# def predict_image(
#     model_rdf: Annotated[
#         Path, typer.Argument(help="Path to the model resource description file (rdf.yaml) or zipped model.")
#     ],
#     inputs: Annotated[List[Path], typer.Option(help="Path(s) to the model input(s).")],
#     outputs: Annotated[List[Path], typer.Option(help="Path(s) for saveing the model output(s).")],
#     # NOTE: typer currently doesn't support union types, so we only support boolean here
#     # padding: Optional[Union[str, bool]] = typer.Argument(
#     #     None, help="Padding to apply in each dimension passed as json encoded string."
#     # ),
#     # tiling: Optional[Union[str, bool]] = typer.Argument(
#     #     None, help="Padding to apply in each dimension passed as json encoded string."
#     # ),
#     padding: Annotated[
#         Optional[bool], typer.Option(help="Whether to pad the image to a size suited for the model.")
#     ] = None,
#     tiling: Annotated[Optional[bool], typer.Option(help="Whether to run prediction in tiling mode.")] = None,
#     weight_format: Annotated[Optional[WeightsFormatEnum], typer.Option(help="The weight format to use.")] = None,
#     devices: Annotated[Optional[List[str]], typer.Option(help="Devices for running the model.")] = None,
# ):
#     if isinstance(padding, str):
#         padding = json.loads(padding.replace("'", '"'))
#         assert isinstance(padding, dict)
#     if isinstance(tiling, str):
#         tiling = json.loads(tiling.replace("'", '"'))
#         assert isinstance(tiling, dict)

#     # this is a weird typer bug: default devices are empty tuple although they should be None
#     if devices is None or len(devices) == 0:
#         devices = None

#     prediction.predict_image(
#         model_rdf, inputs, outputs, padding, tiling, None if weight_format is None else weight_format.value, devices
#     )


# predict_image.__doc__ = prediction.predict_image.__doc__


# @app.command()
# def predict_images(
#     model_rdf: Annotated[
#         Path, typer.Argument(help="Path to the model resource description file (rdf.yaml) or zipped model.")
#     ],
#     input_pattern: Annotated[str, typer.Argument(help="Glob pattern for the input images.")],
#     output_folder: Annotated[str, typer.Argument(help="Folder to save the outputs.")],
#     output_extension: Annotated[Optional[str], typer.Argument(help="Optional output extension.")] = None,
#     # NOTE: typer currently doesn't support union types, so we only support boolean here
#     # padding: Optional[Union[str, bool]] = typer.Argument(
#     #     None, help="Padding to apply in each dimension passed as json encoded string."
#     # ),
#     # tiling: Optional[Union[str, bool]] = typer.Argument(
#     #     None, help="Padding to apply in each dimension passed as json encoded string."
#     # ),
#     padding: Annotated[
#         Optional[bool], typer.Option(help="Whether to pad the image to a size suited for the model.")
#     ] = None,
#     tiling: Annotated[Optional[bool], typer.Option(help="Whether to run prediction in tiling mode.")] = None,
#     weight_format: Annotated[Optional[WeightsFormatEnum], typer.Option(help="The weight format to use.")] = None,
#     devices: Annotated[Optional[List[str]], typer.Option(help="Devices for running the model.")] = None,
# ):
#     input_files = glob(input_pattern)
#     input_names = [os.path.split(infile)[1] for infile in input_files]
#     output_files = [os.path.join(output_folder, fname) for fname in input_names]
#     if output_extension is not None:
#         output_files = [f"{os.path.splitext(outfile)[0]}{output_extension}" for outfile in output_files]

#     if isinstance(padding, str):
#         padding = json.loads(padding.replace("'", '"'))
#         assert isinstance(padding, dict)
#     if isinstance(tiling, str):
#         tiling = json.loads(tiling.replace("'", '"'))
#         assert isinstance(tiling, dict)

#     # this is a weird typer bug: default devices are empty tuple although they should be None
#     if len(devices) == 0:
#         devices = None
#     prediction.predict_images(
#         model_rdf,
#         input_files,
#         output_files,
#         padding=padding,
#         tiling=tiling,
#         weight_format=None if weight_format is None else weight_format.value,
#         devices=devices,
#         verbose=True,
#     )


# predict_images.__doc__ = prediction.predict_images.__doc__


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


if __name__ == "__main__":
    app()
