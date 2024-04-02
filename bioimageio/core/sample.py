from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import xarray as xr
from typing_extensions import Self

from bioimageio.core.tensor_block import TensorBlock

from .axis import AxisId, PerAxis
from .block import Block, BlockNumber, TotalNumberOfBlocks, split_shape_into_blocks
from .common import BlockNumber, Halo, HaloLike, PadMode, SliceInfo
from .stat_measures import Stat
from .tensor import PerTensor, Tensor, TensorId


def split_multiple_shapes_into_blocks(
    shapes: PerTensor[PerAxis[int]],
    block_shapes: PerTensor[PerAxis[int]],
    *,
    strides: Optional[PerTensor[PerAxis[int]]] = None,
    halo: PerTensor[PerAxis[HaloLike]],
    pad_mode: PadMode,
    broadcast: bool = False,
) -> Tuple[TotalNumberOfBlocks, Iterable[PerTensor[Block]]]:
    assert not (
        missing := [t for t in block_shapes if t not in shapes]
    ), f"block shape specified for unknown tensors: {missing}"
    assert broadcast or not (
        missing := [t for t in shapes if t not in block_shapes]
    ), f"no block shape specified for {missing} (set `broadcast` to True if these tensors should be repeated for each block)"
    assert not (
        missing := [t for t in halo if t not in block_shapes]
    ), f"`halo` specified for tensors without block shape: {missing}"

    if strides is None:
        strides = {}

    assert not (
        missing := [t for t in strides if t not in block_shapes]
    ), f"`stride` specified for tensors without block shape: {missing}"

    blocks: Dict[TensorId, Iterable[Block]] = {}
    n_blocks: Dict[TensorId, TotalNumberOfBlocks] = {}
    for t in block_shapes:
        n_blocks[t], blocks[t] = split_shape_into_blocks(
            shape=shapes[t],
            block_shape=block_shapes[t],
            halo=halo.get(t, {}),
            stride=strides.get(t),
        )
        assert n_blocks[t] > 0

    unique_n_blocks = set(n_blocks.values())
    n = max(unique_n_blocks)
    if len(unique_n_blocks) == 2 and 1 in unique_n_blocks:
        if not broadcast:
            raise ValueError(
                f"Mismatch for total number of blocks due to unsplit (single block) tensors: {n_blocks}."
                + " Set `broadcast` to True if you want to repeat unsplit (single block) tensors."
            )

        blocks = {
            t: _repeat_single_block(block_gen, n) if n_blocks[t] == 1 else block_gen
            for t, block_gen in blocks.items()
        }
    elif len(unique_n_blocks) != 1:
        raise ValueError(f"Mismatch for total number of blocks: {n_blocks}")

    return n, _aligned_blocks_generator(n, blocks)


def _aligned_blocks_generator(
    n: TotalNumberOfBlocks, blocks: Dict[TensorId, Iterable[Block]]
):
    iterators = {t: iter(gen) for t, gen in blocks.items()}
    for _ in range(n):
        yield {t: next(it) for t, it in iterators.items()}


def _repeat_single_block(block_generator: Iterable[Block], n: TotalNumberOfBlocks):
    round_two = False
    for block in block_generator:
        assert not round_two
        for _ in range(n):
            yield block

        round_two = True


@dataclass
class Sample:
    """A dataset sample"""

    data: Dict[TensorId, Tensor]
    """the sample's tensors"""

    stat: Stat = field(default_factory=dict)
    """sample and dataset statistics"""

    @property
    def sizes(self) -> PerTensor[PerAxis[int]]:
        return {tid: t.sizes for tid, t in self.data.items()}

    def split_into_blocks(
        self,
        tile_sizes: PerTensor[PerAxis[int]],
        halo: PerTensor[PerAxis[HaloLike]],
        pad_mode: PadMode,
    ) -> TiledSample:
        assert not (
            missing := [t for t in tile_sizes if t not in self.data]
        ), f"`tile_sizes` specified for missing tensors: {missing}"
        assert not (
            missing := [t for t in halo if t not in tile_sizes]
        ), f"`halo` specified for tensors without `tile_sizes`: {missing}"

        # any axis not given in `tile_sizes` is treated
        #   as tile size equal to the tensor axis' size
        explicit_tile_sizes = {
            t: {a: tile_sizes.get(t, {}).get(a, s) for a, s in tdata.sizes.items()}
            for t, tdata in self.data.items()
        }

        tensor_ids = tuple(self.data)
        broadcasted_tensors = {
            t: Tensor.from_xarray(d)
            for t, d in zip(
                tensor_ids, xr.broadcast(*(self.data[tt].data for tt in tensor_ids))
            )
        }

        tile_iterators: Dict[
            TensorId, Iterator[Tuple[BlockNumber, Tensor, PerAxis[SliceInfo]]]
        ] = {}

        n_tiles_common = 1
        last_non_trivial: Optional[TensorId] = None
        for t in tensor_ids:
            n_tiles, generator = broadcasted_tensors[t].block(
                block_size=explicit_tile_sizes[t],
                explicit_halo=halo.get(t, {}),
                pad_mode=pad_mode,
            )
            tile_iterators[t] = iter(generator)
            if n_tiles in (1, n_tiles_common):
                pass
            elif n_tiles_common == 1:
                last_non_trivial = t
                n_tiles_common = n_tiles
            else:
                assert last_non_trivial is not None
                mismatch = {
                    last_non_trivial: {
                        "original sizes": self.data[last_non_trivial].sizes,
                        "broadcasted sizes": broadcasted_tensors[
                            last_non_trivial
                        ].sizes,
                        "n_tiles": n_tiles_common,
                    },
                    t: {
                        "original sizes": self.data[t].sizes,
                        "broadcasted sizes": broadcasted_tensors[t].sizes,
                        "n_tiles": n_tiles,
                    },
                }
                raise ValueError(
                    f"broadcasted tensors {last_non_trivial, t} do not tile to the same"
                    + f" number of tiles {n_tiles_common, n_tiles}. Details\n"
                    + pformat(mismatch)
                )

        for i in range(n_tiles_common):
            data: Dict[TensorId, Tensor] = {}
            inner_slice: Dict[TensorId, PerAxis[SliceInfo]] = {}
            for t, iterator in tile_iterators.items():
                tn, tensor_tile, tensor_slice = next(iterator)
                assert tn == i, f"expected tile number {i}, but got {tn}"
                data[t] = tensor_tile
                inner_slice[t] = tensor_slice

            yield Tile(
                data=data,
                inner_slice=inner_slice,
                halo={
                    t: {a: Halo.create(h) for a, h in th.items()}
                    for t, th in halo.items()
                },
                sample_sizes=self.sizes,
                tile_number=i,
                tiles_in_sample=n_tiles_common,
                stat=self.stat,
            )

    @classmethod
    def from_tiles(
        cls, tiles: Iterable[Tile], *, fill_value: float = float("nan")
    ) -> Self:
        # TODO: add `mode: Literal['in-memory', 'to-disk']` or similar to save out of mem samples
        data: PerTensor[Tensor] = {}
        stat: Stat = {}
        for tile in tiles:
            for t, tile_data in tile.inner_data.items():
                if t not in data:
                    axes = cast(Tuple[AxisId], tile_data.dims)
                    data[t] = Tensor(
                        np.full(
                            tuple(tile.sample_sizes[t][a] for a in axes),
                            fill_value,
                            dtype=tile_data.dtype,
                        ),
                        dims=axes,
                    )

                data[t][tile.inner_slice[t]] = tile_data

            stat = tile.stat

        return cls(data=data, stat=stat)


Sample = Union[UntiledSample, TiledSample]
