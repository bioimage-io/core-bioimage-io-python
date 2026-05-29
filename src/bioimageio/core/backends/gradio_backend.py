import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

import numpy as np
from gradio_client import Client, handle_file
from pydantic import BaseModel
from typing_extensions import Self

from bioimageio.core import MemberId
from bioimageio.core.backends._model_adapter import RemoteModelAdapter
from bioimageio.spec import ModelDescr

from .._common_annotations import PerMemberAnno
from .._prediction_pipeline import RemotePredictionPipeline
from ..io import JsonValue, load_stat, save_tensor, serialize_stat
from ..sample import SampleBlock, SampleBlockMeta
from ..tensor import Tensor
from ._description_serializer import DescriptionSerializer as DescriptionSerializer
from ._sample_serializer import SampleSerializer


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
    data: PerMemberAnno[Union[_SerializableBlock, Path]]
    serialized_stat: List[JsonValue]


SerializedSampleBlock = Dict[str, JsonValue]


class GradioSerializer(SampleSerializer[SerializedSampleBlock]):
    @staticmethod
    def serialize_sample_block(sample_block: SampleBlock) -> SerializedSampleBlock:
        handled_members: Dict[MemberId, _SerializableBlock] = {}
        for m, t in sample_block.members.items():
            handled_members[m] = _SerializableBlock.from_tensor(t)

        serializable = _SerializableSampleBlock(
            data=handled_members,
            meta=sample_block.get_meta(),
            serialized_stat=serialize_stat(sample_block.stat),
        )
        serialized = serializable.model_dump(mode="json")
        print("serialized data", {k: list(v) for k, v in serialized["data"].items()})
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


class GradioModelAdapter(RemoteModelAdapter[SerializedSampleBlock]):
    """Model adapter to use the bioimage-io-gradio-runner as a backend for model inference."""

    def __init__(
        self,
        model_description: ModelDescr,
        *,
        bioimageio_gradio_runner_url: str = "http://bioimage-io-bioimage-io-gradio-runner.hf.space",
    ):
        """Initialize the GradioModelAdapter.

        Note:
            - This adapter requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            bioimageio_gradio_runner_url: The URL of a running bioimage-io-gradio-runner instance (default runner might not be availability/compatible).
        """
        super().__init__(model_description, sample_serializer=GradioSerializer())
        self._client = Client(bioimageio_gradio_runner_url)
        self._serialized_model = DescriptionSerializer.serialize_to_string(
            model_description
        )

    def _forward_sample(
        self, serialized_input_sample: Iterable[SerializedSampleBlock]
    ) -> Iterable[SerializedSampleBlock]:
        job = self._client.submit(
            api_name="/predict-sample",
            model=self._serialized_model,
            input_sample=serialized_input_sample,
            blocksize_parameter=-1,
            skip_preprocessing=True,
            skip_postprocessing=True,
        )
        # n_blocks, outputs, output_rois, serialized_stats = job.result()
        yield from job  # pyright: ignore[reportReturnType]

    def _forward_sample_block(
        self, serialized_input_sample_block: SerializedSampleBlock
    ) -> SerializedSampleBlock:
        job = self._client.submit(
            api_name="/predict-sample-block",
            model=self._serialized_model,
            input_sample_block=serialized_input_sample_block,
            skip_preprocessing=True,
            skip_postprocessing=True,
        )
        return job.result()

    def unload(self):
        return super().unload()


class GradioPredictionPipeline(RemotePredictionPipeline[SerializedSampleBlock]):
    """Prediction pipeline to use the bioimage-io-gradio-runner as a fully remote prediction pipeline."""

    def __init__(
        self,
        model_description: ModelDescr,
        *,
        bioimageio_gradio_runner_url: str = "http://bioimage-io-bioimage-io-gradio-runner.hf.space",
    ):
        """Initialize the GradioPredictionPipeline.

        Note:
            - This pipeline requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            bioimageio_gradio_runner_url: The URL of a running bioimage-io-gradio-runner instance (default runner might not be availability/compatible).
        """
        super().__init__(model_description, serializer=GradioSerializer())
        self._client = Client(bioimageio_gradio_runner_url)
        self._serialized_model = DescriptionSerializer.serialize(model_description)

    def predict_sample_block(
        self,
        sample_block: SampleBlock,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        output_blocks = list(
            self._forward_impl(
                self._serializer.serialize_sample_block(sample_block),
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                blocksize_parameter=-1,
            )
        )
        assert len(output_blocks) == 1, (
            "Expected exactly one output block for a single input block"
        )
        return self._serializer.deserialize_sample_block(output_blocks[0])

    def _forward_impl(
        self,
        serialized_input_sample: SerializedSampleBlock,
        *,
        skip_preprocessing: bool = True,
        skip_postprocessing: bool = True,
        blocksize_parameter: int = -1,
    ) -> Iterable[SerializedSampleBlock]:
        job = self._client.submit(
            api_name="/predict-sample",
            model=self._serialized_model,
            input_sample=serialized_input_sample,
            blocksize_parameter=blocksize_parameter,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
        # n_blocks, outputs, output_rois, serialized_stats = job.result()
        yield from job  # pyright: ignore[reportReturnType]
