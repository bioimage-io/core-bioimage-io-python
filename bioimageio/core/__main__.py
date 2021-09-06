import json
import os
from glob import glob

from pathlib import Path
from typing import List, Optional

import typer

from bioimageio.core import __version__, prediction

try:
    from bioimageio.core.weight_converter import torch as torch_converter
except ImportError:
    torch_converter = None

app = typer.Typer()  # https://typer.tiangolo.com/

# TODO merge spec and core CLI, see https://github.com/bioimage-io/python-bioimage-io/issues/87


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
    padding: Optional[bool] = typer.Argument(None, help="Whether to pad the image to a size suited for the model."),
    tiling: Optional[bool] = typer.Argument(None, help="Whether to run prediction in tiling mode."),
    devices: Optional[List[str]] = typer.Argument(None, help="Devices for running the model."),
) -> int:

    if isinstance(padding, str):
        padding = json.loads(padding.replace("'", '"'))
        assert isinstance(padding, dict)
    if isinstance(tiling, str):
        tiling = json.loads(tiling.replace("'", '"'))
        assert isinstance(tiling, dict)

    # this is a weird typer bug: default devices are empty tuple although they should be None
    if len(devices) == 0:
        devices = None
    prediction.predict_image(model_rdf, inputs, outputs, padding, tiling, devices)
    return 0


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
    padding: Optional[bool] = typer.Argument(None, help="Whether to pad the image to a size suited for the model."),
    tiling: Optional[bool] = typer.Argument(None, help="Whether to run prediction in tiling mode."),
    devices: Optional[List[str]] = typer.Argument(None, help="Devices for running the model."),
) -> int:
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
        model_rdf, input_files, output_files, verbose=True, devices=devices, padding=padding, tiling=tiling
    )
    return 0


predict_images.__doc__ = prediction.predict_images.__doc__


if torch_converter is not None:

    @app.command()
    def convert_torch_weights_to_onnx(
        model_rdf: Path = typer.Argument(
            ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
        ),
        output_path: Path = typer.Argument(..., help="Where to save the onnx weights."),
        opset_version: Optional[int] = typer.Argument(12, help="Onnx opset version."),
        use_tracing: bool = typer.Argument(True, help="Whether to use torch.jit tracing or scripting."),
        verbose: bool = typer.Argument(True, help="Verbosity"),
    ) -> int:
        return torch_converter.convert_weights_to_onnx(model_rdf, output_path, opset_version, use_tracing, verbose)

    convert_torch_weights_to_onnx.__doc__ = torch_converter.convert_weights_to_onnx.__doc__

    @app.command()
    def convert_torch_weights_to_torchscript(
        model_rdf: Path = typer.Argument(
            ..., help="Path to the model resource description file (rdf.yaml) or zipped model."
        ),
        output_path: Path = typer.Argument(..., help="Where to save the torchscript weights."),
        use_tracing: bool = typer.Argument(True, help="Whether to use torch.jit tracing or scripting."),
    ) -> int:
        return torch_converter.convert_weights_to_pytorch_script(model_rdf, output_path, use_tracing)

    convert_torch_weights_to_torchscript.__doc__ = torch_converter.convert_weights_to_pytorch_script.__doc__


if __name__ == "__main__":
    print(f"bioimageio.core package version {__version__}")
    app()
