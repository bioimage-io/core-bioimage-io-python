import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import imageio
import numpy as np
import xarray as xr
from tqdm import tqdm

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import load_resource_description
from bioimageio.spec.model.nodes import Model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model resource (ziped package or rdf.yaml)", required=True)
parser.add_argument("-i", "--input", help="input folder with images for prediction", required=True)
parser.add_argument("-o", "--output", help="output folder to save the predictions", required=True)
parser.add_argument("-e", "--output_extension", help="", default=None)
parser.add_argument("--devices", nargs="+", help="Devices to run this model", default=None)
parser.add_argument("--wildcard", default=None, help="glob wildcard to select files in the input folder")
# implement tiling and enable this as well???
# json-encoded dict
parser.add_argument("--padding", type=str, default=None)








def _predict(prediction_pipeline, in_path, out_path, model, padding):
    axes = tuple(model.inputs[0].axes)
    input_, crop = _load_image(in_path, axes, padding)

    res = prediction_pipeline.forward(*input_)
    # NOTE this only makes sense if the model returns the same shape
    if crop is not None:
        res = res[crop]

    axes = tuple(model.outputs[0].axes)
    _save_image(out_path, res.data, axes)


def main():
    args = parser.parse_args()
    model = load_resource_description(Path(args.model))
    assert isinstance(model, Model)
    if len(model.inputs) > 1 or len(model.outputs) > 1:
        raise ValueError("Bulk prediction is only supported for models with a single input and output")

    wildcard = "*" if args.wildcard is None else args.wildcard
    input_files = glob(os.path.join(args.input, wildcard))

    prediction_pipeline = create_prediction_pipeline(bioimageio_model=model, devices=args.devices)
    os.makedirs(args.output, exist_ok=True)

    padding = args.padding
    if padding is not None:
        padding = json.loads(padding.replace("'", '"'))
        assert isinstance(padding, dict)

    for input_ in tqdm(input_files):
        fname, ext = os.path.splitext(os.path.split(input_)[1])
        if args.output_extension is None:
            output = os.path.join(args.output, f"{fname}{ext}")
        else:
            output = os.path.join(args.output, f"{fname}{args.output_extension}")
        _predict(prediction_pipeline, input_, output, model, padding)

    return 0


if __name__ == "__main__":
    sys.exit(main())
