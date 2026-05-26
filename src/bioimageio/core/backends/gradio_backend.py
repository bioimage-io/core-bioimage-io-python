import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pydantic
from gradio_client import Client, handle_file
from numpy.typing import NDArray
from pydantic import TypeAdapter
from typing_extensions import assert_never

from bioimageio.core import (
    AxisId,
    MemberId,
    Sample,
    SampleBlock,
    Tensor,
)
from bioimageio.core.axis import PerAxis
from bioimageio.core.common import PadMode, PerMember
from bioimageio.core.io import save_tensor
from bioimageio.core.model_adapters import ModelAdapter
from bioimageio.core.sample import SampleBlockMeta
from bioimageio.spec import ModelDescr

from ..io import JsonValue, load_stat, serialize_stat

HandledSampleMembers = PerMember[
    Mapping[str, Union[None, bool, str, Mapping[str, str]]]
]
SampleDims = Mapping[MemberId, Sequence[AxisId]]
SampleShape = Mapping[MemberId, PerAxis[int]]


@pydantic.dataclasses.dataclass
class SerializableSampleBlock:
    members: HandledSampleMembers
    meta: SampleBlockMeta
    # dims: SampleDims
    # sample_shape: SampleShape
    # block_meta: PerMember[BlockMeta]
    stat: List[JsonValue]


sample_adapter = TypeAdapter(SerializableSampleBlock)
SerializedSample = JsonValue


def serialize_sample(sample: Sample, *, out_suffix: str = ".npy") -> SerializedSample:
    return serialize_sample_block(sample.as_single_block(), out_suffix=out_suffix)


def serialize_sample_blockwise_for_model(
    sample: Sample,
    *,
    model: ModelDescr,
    blocksize_parameter: int,
    out_suffix: str = ".npy",
) -> Iterable[SerializedSample]:
    """Split a sample into blocks according to the model's input specifications and `blocksize_parameter` and serialize each block."""
    axis_ns = {
        (ipt.id, axis.id): blocksize_parameter
        for ipt in model.inputs
        for axis in ipt.axes
        if hasattr(axis, "size") and axis.size.__class__.__name__ == "ParameterizedSize"
    }
    input_block_shape = model.get_tensor_sizes(axis_ns, batch_size=1).inputs
    halo = {
        ipt.id: {axis.id: getattr(axis, "halo", 0) for axis in ipt.axes}
        for ipt in model.inputs
    }

    yield from serialize_sample_blockwise(
        sample,
        block_shapes=input_block_shape,
        halo=halo,
        out_suffix=out_suffix,
        pad_mode={ipt.id: ipt.pad or "symmetric" for ipt in model.inputs},
    )


def serialize_sample_blockwise(
    sample: Sample,
    *,
    block_shapes: PerMember[PerAxis[int]],
    halo: PerMember[PerAxis[int]],
    pad_mode: Union[PadMode, PerMember[PadMode]] = "symmetric",
    out_suffix: str = ".npy",
) -> Iterable[SerializedSample]:

    _n_blocks, input_blocks = sample.split_into_blocks(
        block_shapes=block_shapes,
        halo=halo,
        pad_mode=pad_mode,
    )
    for block in input_blocks:
        yield serialize_sample_block(block, out_suffix=out_suffix)


def serialize_sample_block(
    sample: SampleBlock,
    *,
    out_suffix: str = ".npy",
) -> SerializedSample:
    handled_members: HandledSampleMembers = {}
    rois = {}
    for m in sample.members:
        with tempfile.NamedTemporaryFile(
            suffix=_normalize_suffix(out_suffix), delete=False
        ) as tmp:
            if isinstance(sample, SampleBlock):
                t = sample.blocks[m].inner_data
                rois[m] = dict(sample.blocks[m].inner_slice)
            else:
                t = sample.members[m]

            save_tensor(tmp.name, t)
            handled_members[m] = handle_file(Path(tmp.name))

    if isinstance(sample, SampleBlock):
        rois = {k: dict(v.inner_slice) for k, v in sample.blocks.items()}

    output_stats = serialize_stat(sample.stat)
    serializable = SerializableSampleBlock(
        members=handled_members,
        meta=sample.get_meta(),
        stat=output_stats,
    )
    serialized = sample_adapter.dump_python(serializable, mode="json")
    print("serialized", serialized)

    return serialized


def deserialize_sample(
    serialized: SerializedSample,
    output_sample: Optional[Sample] = None,
) -> Sample:
    print("deserializing", serialized)
    deserializable_sample = sample_adapter.validate_python(serialized)
    meta = deserializable_sample.meta
    members: PerMember[Tensor] = {}
    for k, v in deserializable_sample.members.items():
        if "path" not in v:
            raise ValueError(
                f"Invalid handled sample member for member '{k}': {v}. Expected a dict with a 'path' key."
            )
        if not isinstance(v["path"], str):
            raise ValueError(
                f"Invalid path for member '{k}': {v['path']}. Expected a string."
            )
        if not v["path"].endswith(".npy"):
            raise NotImplementedError(
                f"Unsupported file format for member '{k}': {v['path']}. Only .npy files are supported so far."
            )

        members[k] = Tensor.from_numpy(np.load(v["path"]), dims=list(meta.shape[k]))

    if output_sample is None:
        output_sample = Sample.from_blocks(
            [
                SampleBlock.from_meta(
                    meta, data=members, stat=load_stat(deserializable_sample.stat)
                )
            ]
        )
    else:
        for m in output_sample.members:
            output_sample.members[m][meta.blocks[m].inner_slice] = members[m][
                meta.blocks[m].local_slice
            ]

    return output_sample


class GradioModelAdapter(ModelAdapter):
    """Model adapter to use the bioimage-io-gradio-runner as a backend for model inference."""

    def __init__(
        self,
        model_description: ModelDescr,
        bioimageio_gradio_runner_url: str = "http://bioimage-io-bioimage-io-gradio-runner.hf.space",
    ):
        """Initialize the GradioModelAdapter.

        Note:
            - This adapter requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            bioimageio_gradio_runner_url: The URL of a running bioimage-io-gradio-runner instance (default runner might not be availability/compatible).

        """
        super().__init__(model_description)
        self._client = Client(bioimageio_gradio_runner_url)

    def forward(self, input_sample: Union[Sample, SampleBlock]) -> Sample:

        if isinstance(input_sample, Sample):
            serialized_input_sample = serialize_sample(input_sample)
        elif isinstance(input_sample, SampleBlock):
            serialized_input_sample = serialize_sample_block(input_sample)
        else:
            assert_never(input_sample)

        job = self._client.submit(
            api_name="/predict",
            model="bioimage-io/affable-shark",
            input_sample=serialized_input_sample,
            out_suffix=".npy",
            # blocksize_parameter=10,
            # skip_preprocessing = True,
            # skip_postprocessing = True
        )
        # n_blocks, outputs, output_rois, serialized_stats = job.result()

        sample = None
        for serialized_output_sample in job:
            sample = deserialize_sample(serialized_output_sample, output_sample=sample)

        if sample is None:
            raise RuntimeError("No output sample received from the gradio runner.")

        return sample

    def _forward_impl(
        self, input_arrays: Sequence[Optional[NDArray[Any]]]
    ) -> Union[List[Optional[NDArray[Any]]], Tuple[Optional[NDArray[Any]], ...]]:
        raise NotImplementedError(
            "GradioModelAdapter does not support direct forwarding of arrays."
        )

    def unload(self):
        return super().unload()


def _normalize_suffix(suffix: str) -> str:
    normalized = suffix.strip().lower()
    if not normalized:
        return ".npy"
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized
