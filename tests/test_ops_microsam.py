import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("skimage")

from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor


@pytest.fixture(scope="module")
def tid():
    return MemberId("labels")


def _two_cell_maps(shape: "tuple[int, int]" = (64, 64)) -> NDArray[np.float32]:
    """Synthetic AIS decoder maps for two round cells, as (batch, channel, y, x)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    centers = [(16, 16), (16, 48)]
    radius = 16.0

    foreground = np.zeros(shape, dtype="float32")
    center_distances = np.ones(shape, dtype="float32")
    for cy, cx in centers:
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        foreground = np.maximum(foreground, (d < radius).astype("float32"))
        center_distances = np.minimum(center_distances, np.clip(d / radius, 0, 1))

    # inverted boundary distance: low deep inside the cells, high toward the
    # rim, with an explicit ridge at the touching border
    boundary_distances = center_distances.copy()
    boundary_distances[:, 31:33] = 1.0
    return np.stack([foreground, center_distances, boundary_distances])[None]


def _segment(tid: MemberId, maps: NDArray[np.float32], **kwargs: object) -> NDArray[np.uint16]:
    from bioimageio.core.proc_ops import MicroSamWatershed

    sample = Sample(
        members={
            tid: Tensor.from_numpy(maps, dims=[AxisId(a) for a in ("batch", "channel", "y", "x")])
        },
        stat={},
        id=None,
    )
    MicroSamWatershed(labels_id=tid, **kwargs)(sample)  # pyright: ignore[reportArgumentType]
    return sample.members[tid].to_numpy()


def test_two_instances(tid: MemberId):
    seg = _segment(tid, _two_cell_maps())
    assert seg.shape == (1, 1, 64, 64)
    assert seg.dtype == np.uint16

    labels = seg[0, 0]
    ids = set(np.unique(labels)) - {0}
    assert len(ids) == 2
    # instances are separated left/right
    assert labels[16, 16] != labels[16, 48]
    assert labels[16, 16] != 0 and labels[16, 48] != 0
    # background stays background
    assert labels[60, 32] == 0


def test_min_size_filters_small_instances(tid: MemberId):
    maps = _two_cell_maps()
    # add a tiny speck
    maps[0, 0, 58:62, 58:62] = 1.0
    maps[0, 1, 58:62, 58:62] = 0.0
    maps[0, 2, 58:62, 58:62] = 0.0

    unfiltered = _segment(tid, maps, foreground_smoothing=0, distance_smoothing=0)
    filtered = _segment(
        tid, maps, foreground_smoothing=0, distance_smoothing=0, min_size=50
    )
    assert len(set(np.unique(unfiltered)) - {0}) == 3
    assert len(set(np.unique(filtered)) - {0}) == 2
    assert (filtered[0, 0, 58:62, 58:62] == 0).all()


def test_from_proc_descr(tid: MemberId):
    from bioimageio.spec.model.v0_5 import MicroSamWatershedDescr, MicroSamWatershedKwargs

    from bioimageio.core.proc_ops import MicroSamWatershed

    op = MicroSamWatershed.from_proc_descr(
        MicroSamWatershedDescr(
            kwargs=MicroSamWatershedKwargs(min_size=25, output_dtype="uint32")
        ),
        tid,
    )
    assert op.labels_id == tid
    assert op.min_size == 25
    assert op.output_dtype == "uint32"
    assert op.distance_smoothing == 1.6


def test_rejects_wrong_channel_count(tid: MemberId):
    with pytest.raises(ValueError, match="3 stacked tensors"):
        _ = _segment(tid, _two_cell_maps()[:, :2])
