from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "img.png",
        "img.tiff",
        "img.npy",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 5),
        (3, 4, 5),
        (1, 4, 5),
        (5, 4, 3),
        (5, 3, 4),
    ],
)
def test_tensor_io(name: str, shape: Tuple[int, ...], tmp_path: Path):
    from bioimageio.core import Tensor
    from bioimageio.core.io import load_tensor, save_tensor

    path = tmp_path / name
    data = Tensor.from_numpy(
        np.arange(np.prod(shape), dtype=np.uint8).reshape(shape), dims=None
    )
    save_tensor(path, data)
    actual = load_tensor(path)
    assert actual == data
