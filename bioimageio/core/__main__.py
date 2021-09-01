import json
import os
from glob import glob

from pathlib import Path
from typing import List, Optional

import typer

from bioimageio.core import __version__, prediction

app = typer.Typer()  # https://typer.tiangolo.com/

# TODO move weight converter command line utils here
# TODO merge spec and core CLI, see https://github.com/bioimage-io/python-bioimage-io/issues/87


# TODO add support for tiling
# TODO support models with multiple in/outputs
@app.command()
def predict_image(
    model_rdf: Path = typer.Argument(
        ...,
        help="Path to the model resource description file (rdf.yaml) or zipped model."
    ),
    inputs: List[Path] = typer.Argument(..., help="Path(s) to the model input(s)."),
    outputs: List[Path] = typer.Argument(..., help="Path(s) for saveing the model output(s)."),
    padding: Optional[str] = typer.Argument(
        None,
        help="Padding to apply in each dimension passed as json encoded string."
    ),
    devices: Optional[List[str]] = typer.Argument(
        None,
        help="Devices for running the model."
    )
) -> int:
    if padding is not None:
        padding = json.loads(padding.replace("'", "\""))
        assert isinstance(padding, dict)
    prediction.predict_image(model_rdf, inputs, outputs, padding, devices)
    return 0


predict_image.__doc__ = prediction.predict_image.__doc__


# TODO add support for tiling
@app.command()
def predict_images(
    model_rdf: Path = typer.Argument(
        ...,
        help="Path to the model resource description file (rdf.yaml) or zipped model."
    ),
    input_pattern: str = typer.Argument(..., help="Glob pattern for the input images."),
    output_folder: str = typer.Argument(..., help="Folder to save the outputs."),
    output_extension: Optional[str] = typer.Argument(None, help="Optional output extension."),
    padding: Optional[str] = typer.Argument(
        None,
        help="Padding to apply in each dimension passed as json encoded string."
    ),
    devices: Optional[List[str]] = typer.Argument(
        None,
        help="Devices for running the model."
    )
) -> int:
    input_files = glob(input_pattern)
    input_names = [os.path.split(infile)[1] for infile in input_files]
    output_files = [os.path.join(output_folder, fname) for fname in input_names]
    if output_extension is not None:
        output_files = [
            f"{os.path.splitext(outfile)[0]}{output_extension}" for outfile in output_files
        ]

    if padding is not None:
        padding = json.loads(padding.replace("'", "\""))
        assert isinstance(padding, dict)

    prediction.predict_images(input_files, output_files, verbose=True,
                              devices=devices, padding=padding)
    return 0


predict_images.__doc__ = prediction.predict_images.__doc__


if __name__ == "__main__":
    print(f"bioimageio.core package version {__version__}")
    app()
