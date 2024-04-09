import itertools
from dataclasses import dataclass, field
from math import prod
from typing import (
    Any,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from typing_extensions import Self

from .axis import AxisId, PerAxis
from .common import (
    BlockNumber,
    Halo,
    HaloLike,
    MemberId,
    PadWidth,
    PerMember,
    SliceInfo,
    TotalNumberOfBlocks,
)


@dataclass
class LinearAxisTransform:
    axis: AxisId
    scale: float
    offset: int

    def compute(self, s: int) -> int:
        return int(s * self.scale) + self.offset


@dataclass(frozen=True)
class BlockMeta:
    """Block meta data of a sample member (a tensor in a sample)

    Figure for illustration:
    The first 2d block (dashed) of a sample member (**bold**).
    The inner slice (thin) is expanded by a halo in both dimensions on both sides.
    The outer slice reaches from the sample member origin (0, 0) to the right halo point.

    ```terminal
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ─ ─ ─ ─ ─ ─ ─ ┐
    ╷ halo(left)                         ╷
    ╷                                    ╷
    ╷  (0, 0)┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━┯━━━➔
    ╷        ┃                 │         ╷  sample member
    ╷        ┃      inner      │         ╷
    ╷        ┃   (and outer)   │  outer  ╷
    ╷        ┃      slice      │  slice  ╷
    ╷        ┃                 │         ╷
    ╷        ┣─────────────────┘         ╷
    ╷        ┃   outer slice             ╷
    ╷        ┃               halo(right) ╷
    └ ─ ─ ─ ─┃─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
             ⬇
    ```

    note:
    - Inner and outer slices are specified in sample member coordinates.
    - The outer_slice of a block at the sample edge may overlap by more than the
        halo with the neighboring block (the inner slices will not overlap though).

    """

    sample_shape: PerAxis[int]
    """the axis sizes of the whole (unblocked) sample"""

    inner_slice: PerAxis[SliceInfo]
    """inner region (without halo) wrt the sample"""

    halo: PerAxis[Halo]
    """halo enlarging the inner region to the block's sizes"""

    block_number: BlockNumber
    """the n-th block of the sample"""

    blocks_in_sample: TotalNumberOfBlocks
    """total number of blocks in the sample"""

    shape: PerAxis[int] = field(init=False)
    """axis lengths of the block"""

    padding: PerAxis[PadWidth] = field(init=False)
    """padding to realize the halo at the sample edge
    where we cannot simply enlarge the inner slice"""

    outer_slice: PerAxis[SliceInfo] = field(init=False)
    """slice of the outer block (without padding) wrt the sample"""

    inner_shape: PerAxis[int] = field(init=False)
    """axis lengths of the inner region (without halo)"""

    local_slice: PerAxis[SliceInfo] = field(init=False)
    """inner slice wrt the block, **not** the sample"""

    @property
    def dims(self) -> Collection[AxisId]:
        return set(self.inner_shape)

    @property
    def tagged_shape(self) -> PerAxis[int]:
        """alias for shape"""
        return self.shape

    @property
    def inner_slice_wo_overlap(self):
        """subslice of the inner slice, such that all `inner_slice_wo_overlap` can be
        stiched together trivially to form the original sample.

        This can also be used to calculate statistics
        without overrepresenting block edge regions."""
        # TODO: update inner_slice_wo_overlap when adding block overlap
        return self.inner_slice

    def __post_init__(self):
        assert all(
            a in self.sample_shape for a in self.inner_slice
        ), "block has axes not present in sample"
        assert all(
            a in self.inner_slice for a in self.halo
        ), "halo has axes not present in block"

        object.__setattr__(  # TODO: write as property
            self,
            "shape",
            {
                a: s.stop - s.start + (sum(self.halo[a]) if a in self.halo else 0)
                for a, s in self.inner_slice.items()
            },
        )
        assert all(
            s <= self.sample_shape[a] for a, s in self.shape.items()
        ), "block larger than sample"

        object.__setattr__(  # TODO: write as property
            self,
            "inner_shape",
            {a: s.stop - s.start for a, s in self.inner_slice.items()},
        )
        object.__setattr__(  # TODO: write as property
            self,
            "outer_slice",
            {
                a: SliceInfo(
                    max(
                        0,
                        min(
                            self.inner_slice[a].start
                            - (self.halo[a].left if a in self.halo else 0),
                            self.sample_shape[a]
                            - self.inner_shape[a]
                            - (self.halo[a].left if a in self.halo else 0),
                        ),
                    ),
                    min(
                        self.sample_shape[a],
                        self.inner_slice[a].stop
                        + (self.halo[a].right if a in self.halo else 0),
                    ),
                )
                for a in self.inner_slice
            },
        )
        object.__setattr__(  # TODO: write as property
            self,
            "padding",
            {
                a: PadWidth(
                    max(
                        0,
                        self.halo[a].left
                        - (self.inner_slice[a].start + self.outer_slice[a].start),
                    ),
                    max(
                        0,
                        self.halo[a].right
                        - (self.outer_slice[a].stop + self.inner_slice[a].stop),
                    ),
                )
                for a in self.inner_slice
            },
        )
        object.__setattr__(  # TODO: write as property
            self,
            "local_slice",
            {
                a: SliceInfo(
                    self.padding[a].left,
                    self.padding[a].left + self.inner_shape[a],
                )
                for a in self.inner_slice
            },
        )

    def get_transformed(
        self, new_axes: PerAxis[Union[LinearAxisTransform, int]]
    ) -> Self:
        return self.__class__(
            sample_shape={
                a: (
                    trf
                    if isinstance(trf, int)
                    else trf.compute(self.sample_shape[trf.axis])
                )
                for a, trf in new_axes.items()
            },
            inner_slice={
                a: (
                    SliceInfo(0, trf)
                    if isinstance(trf, int)
                    else SliceInfo(
                        trf.compute(self.inner_slice[trf.axis].start),
                        trf.compute(self.inner_slice[trf.axis].stop),
                    )
                )
                for a, trf in new_axes.items()
            },
            halo={
                a: (
                    Halo(0, 0)
                    if isinstance(trf, int)
                    else Halo(self.halo[trf.axis].left, self.halo[trf.axis].right)
                )
                for a, trf in new_axes.items()
            },
            block_number=self.block_number,
            blocks_in_sample=self.blocks_in_sample,
        )


def split_shape_into_blocks(
    shape: PerAxis[int],
    block_shape: PerAxis[int],
    halo: PerAxis[HaloLike],
    stride: Optional[PerAxis[int]] = None,
) -> Tuple[TotalNumberOfBlocks, Generator[BlockMeta, Any, None]]:
    assert all(a in shape for a in block_shape), (
        tuple(shape),
        set(block_shape),
    )
    assert all(a in shape for a in halo), (tuple(shape), set(halo))

    # fill in default halo (0) and block axis length (from tensor shape)
    halo = {a: Halo.create(halo.get(a, 0)) for a in shape}
    block_shape = {a: block_shape.get(a, s) for a, s in shape.items()}
    if stride is None:
        stride = {}

    inner_1d_slices: Dict[AxisId, List[SliceInfo]] = {}
    for a, s in shape.items():
        inner_size = block_shape[a] - sum(halo[a])
        stride_1d = stride.get(a, inner_size)
        inner_1d_slices[a] = [
            SliceInfo(min(p, s - inner_size), min(p + inner_size, s))
            for p in range(0, s, stride_1d)
        ]

    n_blocks = prod(map(len, inner_1d_slices.values()))

    return n_blocks, _block_meta_generator(
        shape,
        blocks_in_sample=n_blocks,
        inner_1d_slices=inner_1d_slices,
        halo=halo,
    )


def _block_meta_generator(
    sample_shape: PerAxis[int],
    *,
    blocks_in_sample: int,
    inner_1d_slices: Dict[AxisId, List[SliceInfo]],
    halo: PerAxis[HaloLike],
):
    assert all(a in sample_shape for a in halo)

    halo = {a: Halo.create(halo.get(a, 0)) for a in inner_1d_slices}
    for i, nd_tile in enumerate(itertools.product(*inner_1d_slices.values())):
        inner_slice: PerAxis[SliceInfo] = dict(zip(inner_1d_slices, nd_tile))

        yield BlockMeta(
            sample_shape=sample_shape,
            inner_slice=inner_slice,
            halo=halo,
            block_number=i,
            blocks_in_sample=blocks_in_sample,
        )


def split_multiple_shapes_into_blocks(
    shapes: PerMember[PerAxis[int]],
    block_shapes: PerMember[PerAxis[int]],
    *,
    halo: PerMember[PerAxis[HaloLike]],
    strides: Optional[PerMember[PerAxis[int]]] = None,
    broadcast: bool = False,
) -> Tuple[TotalNumberOfBlocks, Iterable[PerMember[BlockMeta]]]:
    assert not (
        missing := [t for t in block_shapes if t not in shapes]
    ), f"block shape specified for unknown tensors: {missing}"
    if not block_shapes:
        block_shapes = shapes

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

    blocks: Dict[MemberId, Iterable[BlockMeta]] = {}
    n_blocks: Dict[MemberId, TotalNumberOfBlocks] = {}
    for t in block_shapes:
        n_blocks[t], blocks[t] = split_shape_into_blocks(
            shape=shapes[t],
            block_shape=block_shapes[t],
            halo=halo.get(t, {}),
            stride=strides.get(t),
        )
        assert n_blocks[t] > 0

    assert len(blocks) > 0, blocks
    assert len(n_blocks) > 0, n_blocks
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
    n: TotalNumberOfBlocks, blocks: Dict[MemberId, Iterable[BlockMeta]]
):
    iterators = {t: iter(gen) for t, gen in blocks.items()}
    for _ in range(n):
        yield {t: next(it) for t, it in iterators.items()}


def _repeat_single_block(block_generator: Iterable[BlockMeta], n: TotalNumberOfBlocks):
    round_two = False
    for block in block_generator:
        assert not round_two
        for _ in range(n):
            yield block

        round_two = True
