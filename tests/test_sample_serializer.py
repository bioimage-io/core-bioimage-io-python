from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest

from bioimageio.core._sample_serializer import SampleSerializer
from bioimageio.core._utils.type_hints import NestedList
from bioimageio.core.axis import AxisId
from bioimageio.core.common import MemberId
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec.model import v0_5


class IdentitySerializer(SampleSerializer[Sample]):
    @staticmethod
    def serialize_sample_block(sample_block: Sample) -> Sample:
        return sample_block

    @staticmethod
    def deserialize_sample_block(serialized: Sample) -> Sample:
        return serialized


def make_sample(lazy: bool = False) -> Sample:
    array = np.arange(24).reshape(4, 6)
    raw = Tensor(array, dims=[AxisId("y"), AxisId("x")])
    if lazy:
        raw = raw.chunk({AxisId("y"): 2, AxisId("x"): 3})
    return Sample(
        id="sample",
        members={
            MemberId("raw"): raw,
            MemberId("mask"): Tensor(array % 2, dims=[AxisId("y"), AxisId("x")]),
        },
        stat={},
    )


def test_serialize_whole_sample():
    sample = make_sample()
    serialized = IdentitySerializer.serialize_sample(sample)
    assert serialized is sample
    assert IdentitySerializer.deserialize_sample(serialized) is sample


@pytest.mark.parametrize("lazy", [False, True])
def test_fixed_blocking_round_trip(lazy: bool):
    sample = make_sample(lazy)
    for tensor in sample.members.values():
        tensor.name = "calibrated image"
        tensor.set_physical_scale({AxisId("y"): 0.25, AxisId("x"): 0.2})
        tensor.set_physical_scale_unit({AxisId("y"): "um", AxisId("x"): "um"})
    serialized = IdentitySerializer.serialize_sample_with_fixed_blocking(
        sample,
        block_shapes={m: {AxisId("y"): 2, AxisId("x"): 3} for m in sample.members},
        halo={},
    )
    assert isinstance(serialized, list)
    assert len(serialized) == 2
    for row in serialized:
        assert isinstance(row, list)
        assert len(row) == 2
        for block in row:
            assert isinstance(block, Sample)
            assert block.shape == {
                m: {AxisId("y"): 2, AxisId("x"): 3} for m in sample.members
            }

    restored = IdentitySerializer.deserialize_sample(serialized)
    assert restored.id == sample.id
    assert restored.stat is sample.stat
    for m, tensor in sample.members.items():
        assert restored.members[m].name == tensor.name
        assert restored.members[m].dims == tensor.dims
        assert restored.members[m].get_physical_scale() == tensor.get_physical_scale()
        assert (
            restored.members[m].get_physical_scale_unit()
            == tensor.get_physical_scale_unit()
        )
        np.testing.assert_array_equal(restored.members[m].to_numpy(), tensor.to_numpy())
    assert isinstance(restored.members[MemberId("raw")].data.data, da.Array) == lazy
    assert isinstance(restored.members[MemberId("mask")].data.data, np.ndarray)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("block_y, block_x", [(2, 3), (3, 4)])
def test_model_blocking(lazy: bool, block_y: int, block_x: int):
    sample = make_sample(lazy)
    block_shapes = {
        m: {AxisId("y"): block_y, AxisId("x"): block_x} for m in sample.members
    }
    model = Mock(spec=v0_5.ModelDescr)
    model.get_tensor_sizes.return_value = SimpleNamespace(inputs=block_shapes)
    serialized = IdentitySerializer.serialize_sample_blockwise(
        sample, model=cast(v0_5.ModelDescr, model), blocksize_parameter=2, batch_size=3
    )
    model.get_tensor_sizes.assert_called_once_with(2, 3, sample.shape)
    assert isinstance(serialized, list)
    last_row = serialized[-1]
    assert isinstance(last_row, list)
    last_block = last_row[-1]
    assert isinstance(last_block, Sample)
    assert last_block.members[MemberId("raw")].tagged_shape == {
        AxisId("y"): (4 - 1) % block_y + 1,
        AxisId("x"): (6 - 1) % block_x + 1,
    }
    restored = IdentitySerializer.deserialize_sample(serialized)
    for m, tensor in sample.members.items():
        np.testing.assert_array_equal(restored.members[m].to_numpy(), tensor.to_numpy())


def test_fixed_blocking_retains_halo():
    sample = make_sample()
    serialized = IdentitySerializer.serialize_sample_with_fixed_blocking(
        sample,
        block_shapes={m: {AxisId("y"): 4, AxisId("x"): 5} for m in sample.members},
        halo={m: {AxisId("y"): 1, AxisId("x"): 1} for m in sample.members},
        pad_mode="reflect",
    )
    assert isinstance(serialized, list)
    for y, row in enumerate(serialized):
        assert isinstance(row, list)
        for x, block in enumerate(row):
            assert isinstance(block, Sample)
            for m, tensor in sample.members.items():
                padded = np.pad(tensor.to_numpy(), 1, mode="reflect")
                np.testing.assert_array_equal(
                    block.members[m].to_numpy(),
                    padded[y * 2 : y * 2 + 4, x * 3 : x * 3 + 5],
                )


def test_fixed_blocking_retains_edge_overlap():
    member = MemberId("raw")
    sample = Sample(
        id="sample", members={member: Tensor(np.arange(5), dims=[AxisId("x")])}, stat={}
    )
    serialized = IdentitySerializer.serialize_sample_with_fixed_blocking(
        sample, block_shapes={member: {AxisId("x"): 3}}, halo={}
    )
    assert isinstance(serialized, list)
    assert len(serialized) == 2
    first, last = serialized
    assert isinstance(first, Sample)
    assert isinstance(last, Sample)
    np.testing.assert_array_equal(first.members[member].to_numpy(), [0, 1, 2])
    np.testing.assert_array_equal(last.members[member].to_numpy(), [2, 3, 4])


@pytest.mark.parametrize("serialized", [[], [[]]])
def test_empty_grid(serialized: NestedList[Sample]):
    with pytest.raises(ValueError, match="empty block grid"):
        _ = IdentitySerializer.deserialize_sample(serialized)


@pytest.mark.parametrize("mismatch", ["id", "members", "dimensions"])
def test_incompatible_blocks(mismatch: str):
    first = make_sample()
    second = make_sample()
    if mismatch == "id":
        second.id = "another sample"
    elif mismatch == "members":
        _ = second.members.pop(MemberId("mask"))
    else:
        second.members[MemberId("raw")] = Tensor(
            second.members[MemberId("raw")].to_numpy(),
            dims=[AxisId("x"), AxisId("y")],
        )
    with pytest.raises(ValueError, match="different"):
        _ = IdentitySerializer.deserialize_sample([[first, second]])
