from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Iterable, Iterator, Optional, Tuple, cast

import numpy
import xarray as xr
from typing_extensions import Self

from .axis import AxisId, PerAxis
from .common import Halo, HaloLike, PadMode, SliceInfo, TileNumber
from .stat_measures import Stat
from .tensor import PerTensor, Tensor, TensorId
from .tile import Tile

TiledSample = Iterable[Tile]
"""A dataset sample split into tiles"""


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

    def tile(
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
            TensorId, Iterator[Tuple[TileNumber, Tensor, PerAxis[SliceInfo]]]
        ] = {}

        n_tiles_common = 1
        last_non_trivial: Optional[TensorId] = None
        for t in tensor_ids:
            n_tiles, generator = broadcasted_tensors[t].tile(
                tile_size=explicit_tile_sizes[t],
                halo=halo.get(t, {}),
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
                        numpy.full(
                            tuple(tile.sample_sizes[t][a] for a in axes),
                            fill_value,
                            dtype=tile_data.dtype,
                        ),
                        dims=axes,
                        id=t,
                    )

                data[t][tile.inner_slice[t]] = tile_data

            stat = tile.stat

        return cls(data=data, stat=stat)
