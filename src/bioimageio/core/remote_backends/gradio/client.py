from types import MappingProxyType
from typing import Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

from gradio_client import Client
from loguru import logger

from bioimageio.spec import AnyModelDescr, ValidationSummary
from bioimageio.spec.model import v0_4

from ..._description_serializer import DescriptionSerializer as DescriptionSerializer
from ..._model_adapter import RemoteModelAdapter
from ..._prediction_pipeline import IntermediatePrediction, RemotePredictionPipeline
from ..._settings import settings
from ...axis import PerAxis
from ...common import BlocksizeParameter, PerMember
from ...io import JsonValue
from ...sample import Sample, SampleBlock
from ...stat_measures import Measure, MeasureValue
from .serializer import GradioSampleSerializer

SerializedSampleBlock = Dict[str, JsonValue]


class GradioModelAdapter(RemoteModelAdapter[SerializedSampleBlock]):
    """Model adapter to use the bioimage-io-gradio-runner as a backend for model inference."""

    def __init__(
        self,
        model_description: AnyModelDescr,
        *,
        server: Optional[str] = None,
    ):
        """Initialize the GradioModelAdapter.

        Note:
            - This adapter requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            bioimageio_gradio_server_url: The URL of a running bioimage-io-gradio-server instance (default server might not be availability/compatible).
        """
        super().__init__(
            model_description,
            server=server or settings.default_gradio_server,
            sample_serializer=GradioSampleSerializer(),
        )
        self._client = Client(self.server)
        self._serialized_model, self._sha256 = (
            DescriptionSerializer.serialize_to_string_and_hash(model_description)
        )

    def _forward_impl(
        self, serialized_input_sample: Iterable[SerializedSampleBlock]
    ) -> Iterable[SerializedSampleBlock]:
        return _call_predict_api(
            self._client,
            self._serialized_model,
            self._sha256,
            serialized_input_sample,
            blocksize=None,
            skip_preprocessing=True,
            skip_postprocessing=True,
            skip_input_padding=True,
            skip_output_cropping=True,
            batch_size=None,
        )

    def unload(self):
        return super().unload()

    def load_model(self) -> None:
        for model_data in ("", self._serialized_model):
            try:
                result = self._client.submit(
                    api_name="/load_model", model=model_data, sha256=self._sha256
                ).result()
            except Exception as e:
                if model_data:
                    logger.warning(
                        "Failed to load model on server with model_data, error was: {}",
                        len(model_data),
                        e,
                    )
            else:
                if result:
                    break

    def test_model(self) -> Optional[ValidationSummary]:
        for model_data in ("", self._serialized_model):
            try:
                result = self._client.submit(
                    api_name="/test_model", model=model_data, sha256=self._sha256
                ).result()
            except Exception as e:
                if model_data:
                    logger.warning(
                        "Failed to test model on server with model_data, error was: {}",
                        len(model_data),
                        e,
                    )
            else:
                if result:
                    return ValidationSummary.model_validate_json(result)

        return None


class GradioPredictionPipeline(RemotePredictionPipeline):
    """Prediction pipeline to use the bioimage-io-gradio-runner as a fully remote prediction pipeline."""

    def __init__(
        self,
        model_description: AnyModelDescr,
        *,
        server: Optional[str] = None,
        precomputed_statistics: Mapping[Measure, MeasureValue] = MappingProxyType({}),
        default_blocksize_parameter: BlocksizeParameter = 10,
        default_batch_size: int = 1,
    ):
        """
        Note:
            - This pipeline requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            server: The URL or Hugging Face space name of a running bioimageio gradio server instance (Note: default server might not be availabile/compatible!).
        """
        super().__init__(
            model_description,
            server=server or settings.default_gradio_server,
            default_blocksize_parameter=default_blocksize_parameter,
            default_batch_size=default_batch_size,
        )
        self._client = Client(self.server)
        self._serialized_model, self._sha256 = (
            DescriptionSerializer.serialize_to_string_and_hash(model_description)
        )
        self._serializer = GradioSampleSerializer
        self._precomputed_statistics = dict(precomputed_statistics)

    def predict_sample_block(
        self,
        sample_block: SampleBlock,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
    ) -> SampleBlock:
        if isinstance(self._model_descr, v0_4.ModelDescr):
            raise NotImplementedError(
                f"predict_sample_block not implemented for model {self._model_descr.format_version}"
            )
        else:
            assert self._block_transform is not None

        sample_block.stat.update(self._precomputed_statistics)
        output_block = self._serializer.deserialize_sample(
            _call_predict_api(
                self._client,
                self._serialized_model,
                self._sha256,
                serialized_input_sample=self._serializer.serialize_sample(
                    sample_block.as_sample()
                ),
                blocksize=None,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=True,
                skip_output_cropping=True,
                batch_size=self._default_batch_size,
            )
        )
        output_meta = sample_block.get_transformed_meta(self._block_transform)
        return output_meta.with_data(output_block.members, stat=sample_block.stat)

    def predict_sample_without_blocking(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
    ) -> Sample:
        sample.stat.update(self._precomputed_statistics)
        return self._serializer.deserialize_sample(
            _call_predict_api(
                self._client,
                self._serialized_model,
                self._sha256,
                serialized_input_sample=self._serializer.serialize_sample(sample),
                blocksize=None,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=skip_input_padding,
                skip_output_cropping=skip_output_cropping,
                batch_size=self._default_batch_size,
            )
        )

    def predict_sample_with_fixed_blocking_yield_intermediates(
        self,
        sample: Sample,
        input_block_shape: PerMember[PerAxis[int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        fill_value: float = float("nan"),
    ) -> Tuple[int, Iterable[IntermediatePrediction]]:
        sample.stat.update(self._precomputed_statistics)

        # blocking for serialization is not really important, but we might as well block
        # the same way we want the backend to block for blockwise prediction
        serialized_input_sample = self._serializer.serialize_sample_with_fixed_blocking(
            sample, block_shapes=input_block_shape, halo=self._default_input_halo
        )

        def _predict_blocks():
            output_sample = None
            for serialized_output_block in _call_predict_api(
                self._client,
                self._serialized_model,
                self._sha256,
                serialized_input_sample=serialized_input_sample,
                blocksize=input_block_shape,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=False,
                skip_output_cropping=False,
                batch_size=self._default_batch_size,
            ):
                output_block = self._serializer.deserialize_sample_block(
                    serialized_output_block
                )
                if output_sample is None:
                    output_sample = Sample.from_blocks(
                        [output_block], fill_value=fill_value
                    )
                else:
                    output_sample.set_block(output_block)

                yield IntermediatePrediction(output_sample, output_block)

        block_iterator = _predict_blocks()
        first_intermediate = next(block_iterator)

        def _intermediate_predictions() -> Iterable[IntermediatePrediction]:
            yield first_intermediate
            yield from block_iterator

        return (
            first_intermediate.last_block.blocks_in_sample,
            _intermediate_predictions(),
        )


def _call_predict_api(
    client: Client,
    serialized_model: str,
    sha256: str,
    serialized_input_sample: Iterable[SerializedSampleBlock],
    blocksize: Optional[
        Union[int, Literal["blockwise_as_serialized"], PerMember[PerAxis[int]]]
    ],
    skip_preprocessing: bool,
    skip_postprocessing: bool,
    skip_input_padding: bool,
    skip_output_cropping: bool,
    batch_size: Optional[int],
) -> Iterable[SerializedSampleBlock]:
    def submit(model: str):
        return client.submit(
            api_name="/predict",
            model=model,
            sha256=sha256,
            input_sample=serialized_input_sample,
            blocksize={
                str(k): {str(kk): vv for kk, vv in v.items()}
                for k, v in blocksize.items()
            }
            if not (blocksize is None or isinstance(blocksize, (int, str)))
            else blocksize,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            skip_input_padding=skip_input_padding,
            skip_output_cropping=skip_output_cropping,
            batch_size=batch_size,
        )

    try_with_model_upload = True
    try:
        job = submit("")
        for block in job:  # pyright: ignore[reportUnknownVariableType]
            yield block  # pyright: ignore[reportReturnType]
            # we got one response, so the model cache was hit...
            try_with_model_upload = False
    except Exception as e:
        # A raised exception on the server seems to simply return an empty response sequence,
        # so this except is likely not triggered at all.
        # Below we retry on empty return value, too.
        if try_with_model_upload:
            logger.warning(
                "Failed to submit job without model upload, trying with model upload, error was: {}",
                e,
            )
        else:
            raise e

    if try_with_model_upload:
        job = submit(serialized_model)
        for block in job:  # pyright: ignore[reportUnknownVariableType]
            yield block  # pyright: ignore[reportReturnType]
