import argparse
import sys

import numpy as np
import xarray as xr

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import load_resource_description
from bioimageio.spec.model.nodes import Model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model resource (ziped package or rdf.yaml)", required=True)
parser.add_argument("images", nargs="+", dest="image(s)", help="image(s) to process (.npy)")
parser.add_argument("-o", "--outputs", nargs="+", dest="output(s)", help="output image(s) (.npy)", required=True)
parser.add_argument("--devices", nargs="+", help="Devices to run this model", default=None)


def main():
    args = parser.parse_args()
    model = load_resource_description(args.model)
    assert isinstance(model, Model)
    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=model, devices=args.devices
    )

    if len(args.images) != len(model.inputs):
        raise ValueError(f"Expected {len(model.inputs)} input images, not {len(args.images)}")

    if len(args.outputs) != len(model.outputs):
        raise ValueError(f"Expected {len(model.outputs)} output images, not {len(args.outputs)}")

    input_tensors = [np.load(ipt) for ipt in args.images]
    tagged_data = [xr.DataArray(ipt_tensor, dims=tuple(ipt.axes))
                   for ipt_tensor, ipt in zip(input_tensors, model.inputs)]
    if len(tagged_data) > 1:
        raise NotImplementedError(len(tagged_data))

    res = prediction_pipeline.forward(*tagged_data)
    np.save(args.output, res.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
