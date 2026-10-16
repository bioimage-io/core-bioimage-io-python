from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, overload

from gradio_client.client import Client
from loguru import logger

from bioimageio.spec import AnyModelDescr, ValidationSummary

from ..._description_serializer import DescriptionSerializer
from ..._model_adapter import RemoteModelAdapter
from ..._prediction_pipeline import RemotePredictionPipeline
from ..._settings import settings
from ..._utils.type_hints import NestedList
from ...axis import AxisSize, PerAxis
from ...common import BlocksizeParameter, PerMember
from ...sample import Sample
from ...stat_measures import Measure, MeasureValue
from .serializer import GradioSampleSerializer, SerializedSampleBlock


class GradioModelAdapter(RemoteModelAdapter[SerializedSampleBlock]):
    """Model adapter to use the bioimage-io-gradio-runner as a backend for model inference."""

    def __init__(self, model_description: AnyModelDescr, *, server: str | None = None):
        """Initialize the GradioModelAdapter.

        Note:
            - This adapter requires an environment with the same gradio version as the one used on the bioimage-io-gradio-runner server.

        Args:
            model_description: The model to run inference with.
            server: The URL of a running bioimage-io-gradio-server instance (default server might not be availability/compatible).
        """
        server = server or settings.gradio_server
        if server is None:
            raise ValueError(
                "No gradio server specified. Please provide a server URL or set the 'BIOIMAGEIO_GRADIO_SERVER' environment variable."
            )

        self._client = Client(server, httpx_kwargs={"timeout": 60})
        self._serialized_model, self._sha256 = (
            DescriptionSerializer.serialize_to_string_and_hash(model_description)
        )
        super().__init__(
            model_description, server=server, sample_serializer=GradioSampleSerializer()
        )

    def _forward_impl(
        self, serialized_input_sample: NestedList[SerializedSampleBlock]
    ) -> NestedList[SerializedSampleBlock]:
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

    def load(self) -> None:
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

    def test(self) -> ValidationSummary | None:
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


T = TypeVar("T")


class _DummyJob(Generic[T]):
    def __init__(self, result: T):
        super().__init__()
        self._result = result

    def result(self, timeout: float | None = None) -> T:
        return self._result


class _DummyClient:
    @overload
    def submit(
        self, api_name: Literal["/get_status"], **kwargs: Any
    ) -> _DummyJob[dict[Literal["message"], str]]: ...
    @overload
    def submit(
        self, api_name: Literal["/predict"], **kwargs: Any
    ) -> _DummyJob[NestedList[SerializedSampleBlock]]: ...
    def submit(
        self, api_name: Literal["/get_status", "/predict"], **kwargs: Any
    ) -> (
        _DummyJob[dict[Literal["message"], str]]
        | _DummyJob[NestedList[SerializedSampleBlock]]
    ):
        if api_name == "/get_status":
            return _DummyJob({"message": "dummy client"})
        elif api_name == "/predict":
            from .server import predict as server_predict

            return _DummyJob(server_predict(**kwargs))
        else:
            raise ValueError(f"Unknown api_name: {api_name}")

    def close(self):
        pass


class GradioPredictionPipeline(RemotePredictionPipeline):
    """Prediction pipeline to use the bioimage-io-gradio-runner as a fully remote prediction pipeline."""

    def __init__(
        self,
        model_description: AnyModelDescr,
        *,
        server: str | None | Literal["run_in_client_process"] = None,
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
        server = server or settings.gradio_server
        if server is None:
            raise ValueError(
                "No gradio server specified. Please provide a server URL or set the 'BIOIMAGEIO_GRADIO_SERVER' environment variable."
            )

        super().__init__(
            model_description,
            server=server,
            default_blocksize_parameter=default_blocksize_parameter,
            default_batch_size=default_batch_size,
        )
        self._client = None

        self._serialized_model, self._sha256 = (
            DescriptionSerializer.serialize_to_string_and_hash(model_description)
        )
        self._serializer = GradioSampleSerializer
        self._precomputed_statistics = dict(precomputed_statistics)

    def load(self) -> None:
        if self._server == "run_in_client_process":
            # run via dummy client (intended for debugging)
            self._client = _DummyClient()
        else:
            self._client = Client(self.server, httpx_kwargs={"timeout": 60})

    def unload(self):
        if self._client is not None:
            logger.info(
                "Closing GradioPredictionPipeline client connection to server {}",
                self.server,
            )
            self._client.close()
            self._client = None

    def _predict_without_blocking_impl(
        self,
        sample: Sample,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        skip_input_padding: bool = False,
        skip_output_cropping: bool = False,
    ) -> Sample:
        if self._client is None:
            raise RuntimeError(
                "GradioPredictionPipeline client is not initialized. "
                + "Please call 'load()' before predicting or use the PredictionPipeline as context manager."
            )
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

    def _predict_with_fixed_blocking_impl(
        self,
        sample: Sample,
        block_shape: PerMember[PerAxis[int | AxisSize]],
        *,
        skip_preprocessing: bool,
        skip_postprocessing: bool,
        skip_input_padding: bool,
        skip_output_cropping: bool,
    ) -> Sample:
        if self._client is None:
            raise RuntimeError(
                "GradioPredictionPipeline client is not initialized. "
                + "Please call 'load()' before predicting or use the PredictionPipeline as context manager."
            )

        sample.stat.update(self._precomputed_statistics)
        return self._serializer.deserialize_sample(
            _call_predict_api(
                self._client,
                self._serialized_model,
                self._sha256,
                serialized_input_sample=self._serializer.serialize_sample(sample),
                blocksize=block_shape,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                skip_input_padding=skip_input_padding,
                skip_output_cropping=skip_output_cropping,
                batch_size=self._default_batch_size,
            )
        )

    def get_status(self) -> dict[Literal["message"], str]:
        """Check the status of the server."""
        if self._client is None:
            raise RuntimeError(
                "GradioPredictionPipeline client is not initialized. "
                + "Please call 'load()' before predicting or use the PredictionPipeline as context manager."
            )

        logger.info(
            "Checking status of GradioPredictionPipeline server {}", self.server
        )
        return self._client.submit(api_name="/get_status").result(timeout=30)


def _call_predict_api(
    client: Client | _DummyClient,
    serialized_model: str,
    sha256: str,
    serialized_input_sample: NestedList[SerializedSampleBlock],
    blocksize: int
    | Literal["blockwise_as_serialized"]
    | PerMember[PerAxis[int | AxisSize]]
    | None,
    skip_preprocessing: bool,
    skip_postprocessing: bool,
    skip_input_padding: bool,
    skip_output_cropping: bool,
    batch_size: int | None,
) -> NestedList[SerializedSampleBlock]:
    if isinstance(blocksize, Mapping):
        serialized_blocksize = {
            str(member): {
                str(axis): {"min": size.min, "max": size.max, "step": size.step}
                if isinstance(size, AxisSize)
                else size
                for axis, size in axes.items()
            }
            for member, axes in blocksize.items()
        }
    else:
        serialized_blocksize = blocksize

    def submit(model: str) -> NestedList[SerializedSampleBlock]:
        return client.submit(
            api_name="/predict",
            model=model,
            sha256=sha256,
            input_sample=serialized_input_sample,
            blocksize=serialized_blocksize,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            skip_input_padding=skip_input_padding,
            skip_output_cropping=skip_output_cropping,
            batch_size=batch_size,
        ).result(timeout=300)

    try:
        return submit("")
    except Exception as e:
        logger.warning(
            "Prediction without model upload failed; retrying with model upload: {}", e
        )
        return submit(serialized_model)
