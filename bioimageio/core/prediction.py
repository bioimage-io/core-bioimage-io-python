"""coming soon"""

# TODO: update
import collections.abc
import os
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pydantic import HttpUrl
from tqdm import tqdm
from typing_extensions import assert_never

from bioimageio.core.digest_spec import get_axes_infos, get_member_id, get_member_ids
from bioimageio.core.stat_measures import Stat
from bioimageio.spec import ResourceDescr, load_description
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import AxisType

from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from .axis import AxisInfo
from .sample import Sample
from .tensor import Tensor


# # simple heuristic to determine suitable shape from min and step
# def _determine_shape(min_shape, step, axes):
#     is3d = "z" in axes
#     min_len = 64 if is3d else 256
#     shape = []
#     for ax, min_ax, step_ax in zip(axes, min_shape, step):
#         if ax in "zyx" and step_ax > 0:
#             len_ax = min_ax
#             while len_ax < min_len:
#                 len_ax += step_ax
#             shape.append(len_ax)
#         else:
#             shape.append(min_ax)
#     return shape
