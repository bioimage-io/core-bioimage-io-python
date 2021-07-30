import argparse
import sys

import numpy as np
import torch.cuda
import xarray as xr

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import import_package

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model package (zip file)", required=True)
parser.add_argument("image", nargs="?", help="image to process (.npy)")
parser.add_argument("-o", "--output", nargs="?", help="output image (.npy)", required=True)


def main():
    args = parser.parse_args()
    bioimageio_model = import_package(args.model)
    prediction_pipeline = create_prediction_pipeline(
        bioimageio_model=bioimageio_model, devices=["cuda" if torch.cuda.is_available() else "cpu"]
    )

    input_tensor = np.load(args.image)
    tagged_data = xr.DataArray(input_tensor, dims=tuple(bioimageio_model.input_axes))
    res = prediction_pipeline.forward(tagged_data)
    np.save(args.output, res.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
