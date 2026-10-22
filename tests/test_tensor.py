from __future__ import annotations

from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from xarray.testing import assert_equal  # pyright: ignore[reportUnknownVariableType]

from bioimageio.core import AxisId, Tensor
from bioimageio.core.tensor import TensorMetaData


def test_set_metadata():
    tensor = Tensor(np.zeros((2, 3, 4)), dims="cyx")
    tensor.set_metadata(
        TensorMetaData(
            name="input",
            channel_names=["first", "second"],
            physical_scale={AxisId("y"): 0.25, AxisId("x"): 0.2},
            physical_scale_unit={AxisId("y"): "um", AxisId("x"): "um"},
        )
    )
    assert tensor.name == "input"
    assert tensor.channel_names == ["first", "second"]
    assert tensor.get_physical_scale() == {"channel": 1.0, "y": 0.25, "x": 0.2}
    assert tensor.get_physical_scale_unit() == {"channel": "", "y": "um", "x": "um"}

    metadata = tensor.get_metadata()
    assert metadata == TensorMetaData(
        name="input",
        channel_names=["first", "second"],
        physical_scale={AxisId("channel"): 1.0, AxisId("y"): 0.25, AxisId("x"): 0.2},
        physical_scale_unit={
            AxisId("channel"): "",
            AxisId("y"): "um",
            AxisId("x"): "um",
        },
    )
    restored = Tensor(np.zeros(tensor.shape), dims=tensor.dims)
    restored.set_metadata(metadata)
    assert restored.get_metadata() == metadata

    tensor.set_metadata(
        TensorMetaData(
            name=None, channel_names=None, physical_scale=None, physical_scale_unit=None
        )
    )
    assert tensor.name is None
    assert tensor.channel_names is None
    assert "physical_scale" not in tensor.data.attrs
    assert "physical_scale_unit" not in tensor.data.attrs


@pytest.mark.parametrize(
    "metadata",
    [
        TensorMetaData(
            name=None,
            channel_names=None,
            physical_scale={AxisId("z"): 0.5},
            physical_scale_unit=None,
        ),
        TensorMetaData(
            name=None,
            channel_names=None,
            physical_scale=None,
            physical_scale_unit={AxisId("z"): "um"},
        ),
    ],
)
def test_set_metadata_validates_axes(metadata: TensorMetaData):
    tensor = Tensor(np.zeros((3, 4)), dims="yx")
    with pytest.raises(ValueError, match="not in tensor axes"):
        tensor.set_metadata(metadata)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        "binary",
        "unary",
        "astype",
        "clip",
        "mean",
        "std",
        "sum",
        "var",
        "quantile",
        "slice",
        "transpose",
        "pad",
        "chunk",
    ],
)
def test_processing_preserves_physical_metadata(operation: str, lazy: bool):
    tensor = Tensor(np.arange(24, dtype=np.float32).reshape(4, 6), dims="yx")
    scale = {AxisId("y"): 0.25, AxisId("x"): 0.2}
    unit = {AxisId("y"): "um", AxisId("x"): "um"}
    tensor.set_physical_scale(scale)
    tensor.set_physical_scale_unit(unit)
    if lazy:
        tensor = tensor.as_dask()
    operations: dict[str, Callable[[Tensor], Tensor]] = {
        "binary": lambda x: 2 * x + 1,
        "unary": lambda x: -x,
        "astype": lambda x: x.astype("float64"),
        "clip": lambda x: x.clip(2, 20),
        "mean": lambda x: x.mean(AxisId("y")),
        "std": lambda x: x.std(AxisId("y")),
        "sum": lambda x: x.sum(AxisId("y")),
        "var": lambda x: x.var(AxisId("y")),
        "quantile": lambda x: x.quantile(0.5, AxisId("y")),
        "slice": lambda x: x[{AxisId("y"): 0}],
        "transpose": lambda x: x.transpose("xy"),
        "pad": lambda x: x.pad({AxisId("x"): 1}),
        "chunk": lambda x: x.chunk(
            {AxisId("y"): 2, AxisId("x"): 3}, halo={AxisId("x"): 1}
        ),
    }
    with xr.set_options(keep_attrs=False):
        actual = operations[operation](tensor)
    assert actual.get_physical_scale() == {a: scale[a] for a in actual.dims}
    assert actual.get_physical_scale_unit() == {a: unit[a] for a in actual.dims}
    assert tensor.get_physical_scale() == scale
    assert tensor.get_physical_scale_unit() == unit


@pytest.mark.parametrize("operation", ["mean", "std", "sum", "var", "quantile"])
@pytest.mark.parametrize("dim", [None, AxisId("y"), [AxisId("y"), AxisId("x")], []])
@pytest.mark.parametrize("lazy", [False, True])
def test_reductions_remove_calibration_of_reduced_axes(
    operation: str, dim: AxisId | list[AxisId] | None, lazy: bool
):
    tensor = Tensor(np.arange(24, dtype=np.float32).reshape(4, 6), dims="yx")
    scale = {AxisId("y"): 0.25, AxisId("x"): 0.2}
    unit = {AxisId("y"): "um", AxisId("x"): "um"}
    tensor.set_physical_scale(scale)
    tensor.set_physical_scale_unit(unit)
    tensor.name = "calibrated input"
    if lazy:
        tensor = tensor.as_dask()
    reductions: dict[str, Callable[[], Tensor]] = {
        "mean": lambda: tensor.mean(dim),
        "std": lambda: tensor.std(dim),
        "sum": lambda: tensor.sum(dim),
        "var": lambda: tensor.var(dim),
        "quantile": lambda: tensor.quantile(0.5, dim),
    }
    actual = reductions[operation]()
    assert actual.data.attrs["physical_scale"] == {a: scale[a] for a in actual.dims}
    assert actual.data.attrs["physical_scale_unit"] == {a: unit[a] for a in actual.dims}
    assert actual.name == tensor.name
    assert tensor.data.attrs["physical_scale"] == scale
    assert tensor.data.attrs["physical_scale_unit"] == unit
    if AxisId("y") not in actual.dims:
        expanded = actual.expand_dims([AxisId("y")])
        assert AxisId("y") not in expanded.get_physical_scale()
        assert AxisId("y") not in expanded.data.attrs["physical_scale_unit"]


def _no_execution(*args: object) -> None:
    pytest.fail("as_dask evaluated a task")


@pytest.mark.parametrize("shape,dims", [((), ()), ((2, 3), ("y", "x"))])
def test_as_dask_preserves_data_and_metadata(
    shape: tuple[int, ...], dims: tuple[str, ...]
):
    data = xr.DataArray(
        np.arange(np.prod(shape), dtype="int16").reshape(shape),
        dims=dims,
        coords={d: np.arange(s) + 10 for d, s in zip(dims, shape)},
        name="input",
        attrs={"unit": "intensity"},
    )
    tensor = Tensor.from_xarray(data)
    with Callback(pretask=_no_execution):
        actual = tensor.as_dask()

    assert isinstance(actual.data.data, da.Array)
    assert isinstance(tensor.data.data, np.ndarray)
    assert actual.dtype == tensor.dtype
    assert actual.dims == tensor.dims
    xr.testing.assert_identical(actual.data, data)


def test_as_dask_preserves_existing_graph_and_chunks():
    tensor = Tensor.from_numpy(np.arange(24).reshape(4, 6), dims=("y", "x")).chunk(
        {AxisId("y"): 2, AxisId("x"): 3}
    )
    with Callback(pretask=_no_execution):
        assert tensor.as_dask() is tensor


@pytest.mark.parametrize(
    "axes",
    [
        "yx",
        "xy",
        "cyx",
        "yxc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxc",
        ("batch", "channel", "x", "y"),
    ],
)
def test_transpose_tensor_2d(axes: Sequence[str]):
    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


@pytest.mark.parametrize(
    "axes",
    [
        "zyx",
        "cyzx",
        "yzixc",
        "bczyx",
        "xyz",
        "xyzc",
        "bzyxtc",
        ("batch", "channel", "x", "y", "z"),
    ],
)
def test_transpose_tensor_3d(axes: Sequence[str]):
    tensor = Tensor.from_numpy(np.random.rand(64, 64, 64), dims=None)
    transposed = tensor.transpose([AxisId(a) for a in axes])
    assert transposed.ndim == len(axes)


def test_crop_and_pad():
    tensor = Tensor.from_xarray(
        xr.DataArray(np.random.rand(10, 20), dims=("x", "y"), name="id")
    )
    padded = tensor.pad({AxisId("x"): 7, AxisId("y"): (3, 3)})
    cropped = padded.crop_to(tensor.sizes)
    assert_equal(tensor.data, cropped.data)


def test_some_magic_ops():
    tensor = Tensor.from_numpy(np.random.rand(256, 256), dims=None)
    assert tensor + 2 == 2 + tensor


@pytest.mark.parametrize("operand_kind", ["scalar", "numpy", "xarray", "tensor"])
def test_inplace_add(operand_kind: str):
    tensor = Tensor.from_numpy(np.arange(6).reshape(2, 3), dims="yx")
    operand = (
        2
        if operand_kind == "scalar"
        else np.full((2, 3), 2)
        if operand_kind == "numpy"
        else xr.DataArray(np.full((2, 3), 2), dims=("y", "x"))
        if operand_kind == "xarray"
        else Tensor.from_numpy(np.full((2, 3), 2), dims="yx")
    )
    original = tensor
    tensor += operand
    assert tensor is original
    np.testing.assert_array_equal(tensor.to_numpy(), np.arange(6).reshape(2, 3) + 2)


def test_shape_attributes():
    tensor = Tensor.from_numpy(np.random.rand(1, 2, 25, 26), dims=None)
    assert tensor.shape_tuple == tensor.shape


def test_transpose_stack_extra_dims_to_batch_creates_multi_index():
    from bioimageio.core.tensor import MultiIndex

    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )

    transposed = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )

    assert transposed.dims == (AxisId("batch"), AxisId("x"))
    assert transposed.sizes[AxisId("batch")] == 6
    assert transposed.sizes[AxisId("x")] == 4
    batch_index = transposed.data.indexes[AxisId("batch")]  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(batch_index, MultiIndex)
    assert tuple(str(level) for level in batch_index.names) == ("original_batch", "z")


def test_unstack_batch_multi_index_roundtrip_after_stack():
    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )

    stacked = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )
    unstacked = stacked.unstack_batch_multi_index()
    restored = unstacked.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="raise",
    )

    np.testing.assert_array_equal(restored.to_numpy(), tensor.to_numpy())


def test_unstack_batch_multi_index_error_modes():
    no_batch = Tensor.from_numpy(np.arange(6).reshape(2, 3), dims=("y", "x"))
    with pytest.raises(ValueError, match="without a 'batch' axis"):
        _ = no_batch.unstack_batch_multi_index()
    assert no_batch.unstack_batch_multi_index(errors="ignore") is no_batch

    non_multi_index_batch = Tensor.from_numpy(
        np.arange(6).reshape(2, 3), dims=("batch", "x")
    )
    with pytest.raises(ValueError, match="does not have a MultiIndex"):
        _ = non_multi_index_batch.unstack_batch_multi_index()
    assert (
        non_multi_index_batch.unstack_batch_multi_index(errors="ignore")
        is non_multi_index_batch
    )


def test_transpose_unstacks_missing_dims_from_batch():
    tensor = Tensor.from_numpy(
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("batch", "z", "x"),
    )
    stacked = tensor.transpose(
        [AxisId("batch"), AxisId("x")],
        extra_dims="stack",
        missing_dims="raise",
    )

    unstacked = stacked.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="unstack",
    )

    assert unstacked.dims == (AxisId("batch"), AxisId("z"), AxisId("x"))
    np.testing.assert_array_equal(unstacked.to_numpy(), tensor.to_numpy())


def test_transpose_unstack_or_expand_expands_without_multi_index_batch():
    tensor = Tensor.from_numpy(np.arange(2 * 4).reshape(2, 4), dims=("batch", "x"))

    transposed = tensor.transpose(
        [AxisId("batch"), AxisId("z"), AxisId("x")],
        extra_dims="raise",
        missing_dims="unstack_or_expand",
    )

    assert transposed.dims == (AxisId("batch"), AxisId("z"), AxisId("x"))
    assert transposed.sizes[AxisId("batch")] == 2
    assert transposed.sizes[AxisId("z")] == 1
    assert transposed.sizes[AxisId("x")] == 4
