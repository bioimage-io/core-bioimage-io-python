"""Run original cellpose model and save an analog input and output for bioimageio tests
Works on cellpose 4.1, but not since cellpose 4.2
"""

import os
from pathlib import Path

import cellpose.models
import imageio
import numpy as np

if __name__ == "__main__":
    # Pin the exact same checkpoint the bioimageio export uses (HuggingFace `cpsam`,
    # sha256 e1440429...). CellposeModel() otherwise defaults to `cpsam_v2`, whose
    # weights differ (347/348 tensors), producing a non-reproducible reference.
    cellpose_original = cellpose.models.CellposeModel(
        gpu=False, pretrained_model="/home/dfranco/.cellpose/models/cpsam"
    )

    os.chdir(Path(__file__).parent)
    input_array = imageio.imread("sample_input.png").transpose(2, 0, 1)
    assert isinstance(input_array, np.ndarray)
    print("input:", input_array.shape)
    input_array = input_array[None, :, 256:512, 256:512]
    print("input roi:", input_array.shape)

    # 16 pixel halo to roughly match the 0.1*256 (tile size) tile overlap used in cellpose
    h = 16
    input_array = input_array[:, :, h:-h, h:-h]
    print("input cropped:", input_array.shape)

    # pad by 8 (additionally) before calling cellpose model (that pads another 8 internally)
    # so that cellpose input has 16 pixels padded (just like the bioimageio package we will export).
    p = 8
    input_array = np.pad(input_array, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    print("input padded:", input_array.shape)

    # add the cellpose internal padding of 8 pixels and save as test input for the bioimageio description
    input_array_padded = np.pad(
        input_array, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant"
    )
    print("test input padded:", input_array_padded.shape)
    Path("output").mkdir(exist_ok=True)
    np.save("output/test_input.npy", input_array_padded)
    imageio.imwrite("output/test_input.tiff", input_array_padded[0].transpose(1, 2, 0))

    # run original cellpose model
    mask, flows, styles = cellpose_original.eval(input_array.transpose(0, 2, 3, 1))
    print("mask shape:", mask.shape)

    # pad output mask by 8 pixels to that were cropped internally by cellpose
    mask = np.pad(mask, ((p, p), (p, p)), mode="constant")

    # save output with batch and channel axes
    print("padded mask shape:", mask.shape)
    np.save("output/test_output.npy", mask[None, None])
    imageio.imwrite("output/test_output.tiff", mask.astype(np.uint16))

    # write out flows for debugging
    flows = flows[0]
    print("flows", flows.shape)
    np.save("output/flows.npy", flows)
    imageio.imwrite("output/flows.tiff", flows)
