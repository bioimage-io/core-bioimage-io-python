import sys
from pathlib import Path
from typing import List, Optional, Union

import fire

from bioimageio.core import __version__, test_description
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.collection import CollectionDescr
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr
from bioimageio.spec.model.v0_5 import WeightsFormat
from bioimageio.spec.notebook import NotebookDescr


class Bioimageio:
    """CLI to work with resources shared on bioimage.io"""

    @staticmethod
    def package(
        source: str,
        path: Path = Path("bioimageio-package.zip"),
        weight_format: Optional[WeightsFormat] = None,
    ):
        """Package a bioimageio resource as a zip file

        Args:
            source: RDF source e.g. `bioimageio.yaml` or `http://example.com/rdf.yaml`
            path: output path
            weight-format: include only this single weight-format
        """
        _ = save_bioimageio_package(
            source,
            output_path=path,
            weights_priority_order=None if weight_format is None else (weight_format,),
        )

    @staticmethod
    def test(
        source: str,
        weight_format: Optional[WeightsFormat] = None,
        *,
        devices: Optional[Union[str, List[str]]] = None,
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
        summary = test_description(
            source,
            weight_format=None if weight_format is None else weight_format,
            devices=[devices] if isinstance(devices, str) else devices,
            decimal=decimal,
        )
        print(f"\ntesting model {source}...")
        print(summary.format())
        sys.exit(0 if summary.status == "passed" else 1)


assert isinstance(Bioimageio.__doc__, str)
Bioimageio.__doc__ += f"""

library versions:
  bioimageio.core {__version__}
  bioimageio.spec {__version__}

spec format versions:
        model RDF {ModelDescr.implemented_format_version}
      dataset RDF {DatasetDescr.implemented_format_version}
     notebook RDF {NotebookDescr.implemented_format_version}
   collection RDF {CollectionDescr.implemented_format_version}

"""

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


def main():
    fire.Fire(Bioimageio, name="bioimageio")


if __name__ == "__main__":
    main()
