from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from gradio_client import handle_file
from pydantic import BaseModel
from typing_extensions import Self

from ..._common_annotations import PerMemberAnno
from ..._sample_serializer import SampleSerializer
from ...common import MemberId
from ...io import JsonValue, load_stat, save_tensor, serialize_stat
from ...sample import SampleBlock, SampleBlockMeta
from ...tensor import Tensor


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


class _SerializableSampleBlock(BaseModel, frozen=True):
    meta: SampleBlockMeta
    data: PerMemberAnno[_SerializableBlock | Path]
    serialized_stat: list[JsonValue]


SerializedSampleBlock = dict[str, JsonValue]


class GradioSampleSerializer(SampleSerializer[SerializedSampleBlock]):
    @staticmethod
    def serialize_sample_block(sample_block: SampleBlock) -> SerializedSampleBlock:
        handled_members: dict[MemberId, _SerializableBlock] = {}
        for m, t in sample_block.members.items():
            handled_members[m] = _SerializableBlock.from_tensor(t)

        serializable = _SerializableSampleBlock(
            data=handled_members,
            meta=sample_block.get_meta(),
            serialized_stat=serialize_stat(sample_block.stat),
        )
        serialized = serializable.model_dump(mode="json")
        return serialized

    @staticmethod
    def deserialize_sample_block(serialized: SerializedSampleBlock) -> SampleBlock:
        deserializable_sample = _SerializableSampleBlock.model_validate(serialized)
        sample_meta = deserializable_sample.meta
        members = {
            k: Tensor.from_numpy(
                np.load(v if isinstance(v, Path) else v.path),
                dims=list(sample_meta.shape[k]),
            )
            for k, v in deserializable_sample.data.items()
        }
        return SampleBlock.from_meta(
            sample_meta,
            data=members,
            stat=load_stat(deserializable_sample.serialized_stat),
        )
