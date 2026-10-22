from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, cast

import numpy as np
from gradio_client import handle_file
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Self

from ..._common_annotations import PerMemberAnno
from ..._sample_serializer import SampleSerializer
from ...axis import AxisId
from ...common import MemberId, SampleId
from ...io import JsonValue, load_stat, save_tensor, serialize_stat
from ...sample import Sample
from ...tensor import MultiIndex, Tensor


class _SerializableBlock(BaseModel, frozen=True):
    path: Path
    meta: Mapping[str, str]
    orig_name: str

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            save_tensor(tmp.name, tensor)

        handled = handle_file(Path(tmp.name))
        return cls.model_validate(handled)


def _restore_sample_id(value: object) -> object:
    # JSON encodes tuple sample IDs as lists.
    if isinstance(value, list):
        return tuple(_restore_sample_id(item) for item in cast(list[object], value))
    return value


class _SerializableSampleBlock(BaseModel, frozen=True):
    id: Annotated[SampleId, BeforeValidator(_restore_sample_id)]
    dims: PerMemberAnno[tuple[AxisId, ...]]
    data: PerMemberAnno[_SerializableBlock | Path]
    serialized_stat: list[JsonValue]
    batch_indices: PerMemberAnno[dict[str, list[JsonValue]]] = {}


SerializedSampleBlock = dict[str, JsonValue]


class GradioSampleSerializer(SampleSerializer[SerializedSampleBlock]):
    @staticmethod
    def serialize_sample_block(sample_block: Sample) -> SerializedSampleBlock:
        handled_members: dict[MemberId, _SerializableBlock] = {}
        for m, t in sample_block.members.items():
            handled_members[m] = _SerializableBlock.from_tensor(t)

        serializable = _SerializableSampleBlock(
            data=handled_members,
            id=sample_block.id,
            dims={m: t.dims for m, t in sample_block.members.items()},
            serialized_stat=serialize_stat(sample_block.stat),
            batch_indices={
                m: {
                    str(name): index.get_level_values(i).tolist()
                    for i, name in enumerate(index.names)
                }
                for m, tensor in sample_block.members.items()
                if (index := tensor.batch_multi_index) is not None
            },
        )
        serialized = serializable.model_dump(mode="json")
        return serialized

    @staticmethod
    def deserialize_sample_block(serialized: SerializedSampleBlock) -> Sample:
        deserializable_sample = _SerializableSampleBlock.model_validate(serialized)
        if deserializable_sample.data.keys() != deserializable_sample.dims.keys():
            raise ValueError(
                "Serialized sample data and dimensions have different members"
            )

        members = {
            k: Tensor(
                np.load(v if isinstance(v, Path) else v.path, allow_pickle=False),
                dims=deserializable_sample.dims[k],
            )
            for k, v in deserializable_sample.data.items()
        }
        for member, index in deserializable_sample.batch_indices.items():
            members[member] = members[member].assign_batch_multi_index(
                MultiIndex.from_arrays(list(index.values()), names=list(index))
            )
        return Sample(
            id=deserializable_sample.id,
            members=members,
            stat=load_stat(deserializable_sample.serialized_stat),
        )
