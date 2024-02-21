# # TODO: update

# from __future__ import annotations

# import os
# from copy import deepcopy
# from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

# import imageio
# import numpy as np
# from numpy.typing import ArrayLike, NDArray
# from xarray import DataArray

# from bioimageio.spec._internal.io_utils import load_array
# from bioimageio.spec.model.v0_4 import InputTensorDescr as InputTensor04
# from bioimageio.spec.model.v0_4 import OutputTensorDescr as OutputTensor04
# from bioimageio.spec.model.v0_5 import InputTensorDescr as InputTensor05
# from bioimageio.spec.model.v0_5 import OutputTensorDescr as OutputTensor05

# InputTensor = Union[InputTensor04, InputTensor05]
# OutputTensor = Union[OutputTensor04, OutputTensor05]


# #
# # helper functions to transform input images / output tensors to the required axes
# #


# def transpose_image(image: NDArray[Any], desired_axes: str, current_axes: Optional[str] = None) -> NDArray[Any]:
#     """Transform an image to match desired axes.

#     Args:
#         image: the input image
#         desired_axes: the desired image axes
#         current_axes: the axes of the input image
#     """
#     # if the image axes are not given deduce them from the required axes and image shape
#     if current_axes is None:
#         has_z_axis = "z" in desired_axes
#         ndim = image.ndim
#         if ndim == 2:
#             current_axes = "yx"
#         elif ndim == 3:
#             current_axes = "zyx" if has_z_axis else "cyx"
#         elif ndim == 4:
#             current_axes = "czyx"
#         elif ndim == 5:
#             current_axes = "bczyx"
#         else:
#             raise ValueError(f"Invalid number of image dimensions: {ndim}")

#     tensor = DataArray(image, dims=tuple(current_axes))
#     # expand the missing image axes
#     missing_axes = tuple(set(desired_axes) - set(current_axes))
#     tensor = tensor.expand_dims(dim=missing_axes)
#     # transpose to the correct axis order
#     tensor = tensor.transpose(*tuple(desired_axes))
#     # return numpy array
#     ret: NDArray[Any] = tensor.values
#     return ret


# #
# # helper functions for loading and saving images
# #


# def load_image(in_path, axes: Sequence[str]) -> DataArray:
#     ext = os.path.splitext(in_path)[1]
#     if ext == ".npy":
#         im = load_array(in_path)
#     else:
#         is_volume = "z" in axes
#         im = imageio.volread(in_path) if is_volume else imageio.imread(in_path)
#         im = transpose_image(im, axes)
#     return DataArray(im, dims=axes)


# def load_tensors(sources, tensor_specs: List[Union[InputTensor, OutputTensor]]) -> List[DataArray]:
#     return [load_image(s, sspec.axes) for s, sspec in zip(sources, tensor_specs)]


# #
# # helper function for padding
# #


# def pad(image, axes: Sequence[str], padding, pad_right=True) -> Tuple[np.ndarray, Dict[str, slice]]:
#     assert image.ndim == len(axes), f"{image.ndim}, {len(axes)}"

#     padding_ = deepcopy(padding)
#     mode = padding_.pop("mode", "dynamic")
#     assert mode in ("dynamic", "fixed")

#     is_volume = "z" in axes
#     if is_volume:
#         assert len(padding_) == 3
#     else:
#         assert len(padding_) == 2

#     if isinstance(pad_right, bool):
#         pad_right = len(axes) * [pad_right]

#     pad_width = []
#     crop = {}
#     for ax, dlen, pr in zip(axes, image.shape, pad_right):
#         if ax in "zyx":
#             pad_to = padding_[ax]

#             if mode == "dynamic":
#                 r = dlen % pad_to
#                 pwidth = 0 if r == 0 else (pad_to - r)
#             else:
#                 if pad_to < dlen:
#                     msg = f"Padding for axis {ax} failed; pad shape {pad_to} is smaller than the image shape {dlen}."
#                     raise RuntimeError(msg)
#                 pwidth = pad_to - dlen

#             pad_width.append([0, pwidth] if pr else [pwidth, 0])
#             crop[ax] = slice(0, dlen) if pr else slice(pwidth, None)
#         else:
#             pad_width.append([0, 0])
#             crop[ax] = slice(None)

#     image = np.pad(image, pad_width, mode="symmetric")
#     return image, crop
