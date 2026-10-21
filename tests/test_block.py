from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pytest

from bioimageio.core.axis import AxisId, AxisSize
from bioimageio.core.block import Block, apply_blockwise, split_tensor_into_blocks
from bioimageio.core.common import Halo, MemberId, PerMember
from bioimageio.core.tensor import Tensor


@pytest.mark.parametrize("pad_boundary", ["allow", True, False])
@pytest.mark.parametrize("halo", [0, 1])
def test_single_block_padding(pad_boundary: Literal["allow", True, False], halo: int):
    x = AxisId("x")
    tensor = Tensor(np.arange(10), dims=[x])
    blocks = split_tensor_into_blocks(
        tensor,
        block_shape={x: AxisSize(min=4, max=16, step=4)},
        halo={x: halo},
        pad_mode="edge",
        pad_boundary=pad_boundary,
    )
    assert blocks.shape == ((2,) if pad_boundary is False else (1,))
    for block in cast(Sequence[Block], blocks.flat):
        assert (block.outer_tensor.tagged_shape[x] - 4) % 4 == 0
        if pad_boundary is False:
            assert not block.padding
    if pad_boundary == "allow":
        np.testing.assert_array_equal(
            blocks[0].outer_tensor.to_numpy(), np.r_[np.arange(10), 9, 9]
        )


@pytest.mark.parametrize("pad_boundary", ["allow", True, False])
@pytest.mark.parametrize("mode", ["parallel", "sequential"])
@pytest.mark.parametrize("halo", [0, 1])
@pytest.mark.parametrize("length", [3, 10, 30])
def test_apply_blockwise_padding(
    pad_boundary: Literal["allow", True, False],
    mode: Literal["parallel", "sequential"],
    halo: int,
    length: int,
):
    if length == 3 and pad_boundary is False:
        pytest.skip("Input is smaller than the minimum valid block")
    x, member = AxisId("x"), MemberId("input")
    tensor = Tensor(np.arange(length), dims=[x])
    calls: list[int] = []

    def identity(inputs: PerMember[Tensor]) -> PerMember[Tensor]:
        size = inputs[member].tagged_shape[x]
        assert 4 <= size <= 16 and (size - 4) % 4 == 0
        calls.append(size)
        return inputs

    outputs = apply_blockwise(
        identity,
        {member: tensor},
        block_shape={member: {x: AxisSize(min=4, max=16, step=4)}},
        chunk_shape={member: {x: length + (2 * halo if pad_boundary is True else 0)}},
        pad_mode="edge",
        pad_boundary=pad_boundary,
        input_halo={member: {x: Halo(halo, halo)}},
        output_halo={member: {x: Halo(halo, halo)}},
        get_output_shape=lambda shapes: shapes,
        blockwise_processing_mode=mode,
    )
    np.testing.assert_array_equal(outputs[member].to_numpy(), tensor.to_numpy())
    if length <= 10 and pad_boundary is not False:
        assert len(calls) == 1


def test_single_block_respects_maximum():
    x = AxisId("x")
    blocks = split_tensor_into_blocks(
        Tensor(np.arange(10), dims=[x]),
        block_shape={x: AxisSize(min=4, max=10, step=4)},
        halo={},
        pad_mode="edge",
        pad_boundary="allow",
    )
    assert blocks.shape == (2,)
    assert all(
        block.outer_tensor.tagged_shape[x] <= 10
        for block in cast(Sequence[Block], blocks.flat)
    )


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("pad_boundary", ["allow", True, False])
def test_padding_with_multiple_chunks_and_scaled_output(
    scale: int, pad_boundary: Literal["allow", True, False]
):
    x, y, member = AxisId("x"), AxisId("y"), MemberId("input")
    data = np.arange(300).reshape(10, 30)
    calls: list[tuple[int, int]] = []

    def repeat(inputs: PerMember[Tensor]) -> PerMember[Tensor]:
        tensor = inputs[member]
        calls.append((tensor.tagged_shape[x], tensor.tagged_shape[y]))
        assert all(4 <= s <= 16 and s % 4 == 0 for s in calls[-1])
        return {
            member: Tensor(np.repeat(tensor.to_numpy(), scale, axis=0), dims=[x, y])
        }

    result = apply_blockwise(
        repeat,
        {member: Tensor(data, dims=[x, y])},
        block_shape={member: {x: AxisSize(4, 16, 4), y: AxisSize(4, 16, 4)}},
        chunk_shape={member: {x: 12 if pad_boundary is True else 10, y: 18}},
        pad_mode={member: "edge"},
        pad_boundary=pad_boundary,
        input_halo={member: {x: Halo(1, 1), y: Halo(1, 1)}},
        output_halo={member: {x: Halo(scale, scale), y: Halo(1, 1)}},
        get_output_shape=lambda shapes: {
            member: {x: shapes[member][x] * scale, y: shapes[member][y]}
        },
    )
    np.testing.assert_array_equal(
        result[member].to_numpy(), np.repeat(data, scale, axis=0)
    )
    if pad_boundary is not False:
        assert all(sx == 12 for sx, _ in calls)


@pytest.mark.parametrize("block_size", [8, AxisSize(8, 16, 4)])
def test_boundary_padding_forbidden_for_small_tensor(block_size: int | AxisSize):
    x = AxisId("x")
    with pytest.raises(ValueError, match="larger than tensor"):
        _ = split_tensor_into_blocks(
            Tensor(np.arange(3), dims=[x]),
            block_shape={x: block_size},
            halo={},
            pad_mode="edge",
            pad_boundary=False,
        )
