from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, overload

from gradio_client.client import Client, Job
from loguru import logger

from bioimageio.spec import AnyModelDescr, ValidationSummary
from bioimageio.spec.model import v0_4

from ..._description_serializer import DescriptionSerializer
from ..._model_adapter import RemoteModelAdapter
from ..._prediction_pipeline import RemotePredictionPipeline
from ..._settings import settings
from ..._utils.type_guards import is_sequence
from ...axis import PerAxis
from ...common import BlocksizeParameter, PerMember
from ...io import JsonValue
from ...sample import Sample
from ...stat_measures import Measure, MeasureValue
from .serializer import GradioSampleSerializer

SerializedSampleBlock = dict[str, JsonValue]


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

    def __iter__(self):
        yield self._result


class _DummyJobIterable(Generic[T]):
    def __init__(self, result: Iterable[T]):
        assert isinstance(result, Iterable), "result must be iterable"
        super().__init__()
        self._count = 0
        self._result = result
        self._outputs: list[T] = []
        self._done = False

    def status(self) -> dict[Literal["status"], str]:
        return {"status": "dummy done" if self._done else "dummy running"}

    def result(self, timeout: float | None = None) -> T:
        self._outputs = list(self)
        return self._outputs[-1]

    def done(self) -> bool:
        return self._done

    def cancel(self) -> bool:
        self._result = ()
        return True

    def outputs(self) -> list[T]:
        return list(self._outputs)

    def __iter__(self) -> Iterator[T]:
        for res in self._outputs:
            yield res

        for res in self._result:
            self._outputs.append(res)
            yield res
        self._done = True


class _DummyClient:
    @overload
    def submit(
        self, api_name: Literal["/get_status"], **kwargs: Any
    ) -> _DummyJob[dict[Literal["message"], str]]: ...
    @overload
    def submit(
        self, api_name: Literal["/predict"], **kwargs: Any
    ) -> _DummyJobIterable[SerializedSampleBlock]: ...
    def submit(
        self, api_name: Literal["/get_status", "/predict"], **kwargs: Any
    ) -> (
        _DummyJob[dict[Literal["message"], str]]
        | _DummyJobIterable[SerializedSampleBlock]
    ):
        if api_name == "/get_status":
            return _DummyJob({"message": "dummy client"})
        elif api_name == "/predict":
            from .server import predict as server_predict

            return _DummyJobIterable(server_predict(**kwargs))
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
        if server == "run_in_client_process":
            # run via dummy client (intended for debugging)
            self._client = _DummyClient()
        else:
            self._client = Client(self.server, httpx_kwargs={"timeout": 60})

        self._serialized_model, self._sha256 = (
            DescriptionSerializer.serialize_to_string_and_hash(model_description)
        )
        self._serializer = GradioSampleSerializer
        self._precomputed_statistics = dict(precomputed_statistics)

    def close(self):
        logger.info(
            "Closing GradioPredictionPipeline client connection to server {}",
            self.server,
        )
        self._client.close()

    def predict_block(
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
                serialized_input_sample=(
                    self._serializer.serialize_sample_block(sample_block),
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

    def _predict_without_blocking_impl(
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

    def _predict_with_fixed_blocking_impl(
        self,
        sample: Sample,
        input_block_shape: PerMember[PerAxis[int]],
        *,
        skip_preprocessing: bool = False,
        skip_postprocessing: bool = False,
        fill_value: float = float("nan"),
    ) -> tuple[int, Iterable[IntermediatePrediction]]:
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
            first_intermediate.last_block.blocks_loc_shape,
            _intermediate_predictions(),
        )

    def get_status(self) -> dict[Literal["message"], str]:
        """Check the status of the server."""
        logger.info(
            "Checking status of GradioPredictionPipeline server {}", self.server
        )
        return self._client.submit(api_name="/get_status").result(timeout=30)


def _call_predict_api(
    client: Client | _DummyClient,
    serialized_model: str,
    sha256: str,
    serialized_input_sample: Iterable[SerializedSampleBlock],
    blocksize: int
    | Literal["blockwise_as_serialized"]
    | PerMember[PerAxis[int]]
    | None,
    skip_preprocessing: bool,
    skip_postprocessing: bool,
    skip_input_padding: bool,
    skip_output_cropping: bool,
    batch_size: int | None,
) -> Iterable[SerializedSampleBlock]:
    def submit(model: str, input_sample: list[SerializedSampleBlock]):
        return client.submit(
            api_name="/predict",
            model=model,
            sha256=sha256,
            input_sample=input_sample,
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

    def iterate_job(
        job: _DummyJobIterable[SerializedSampleBlock] | Job,
    ) -> Iterable[SerializedSampleBlock]:
        seconds_between_polls: float = 1.0
        timeout_seconds: float = 300.0
        output_idx = 0
        started = time.time()
        seconds_to_sleep = 0.01  # first sleep is shorter for faster initial responses
        while not job.done():
            if (time.time() - started) > timeout_seconds:
                _ = job.cancel()
                raise TimeoutError(
                    f"Timeout of {timeout_seconds} seconds exceeded while waiting for prediciton job {job.status()}."
                )

            logger.debug("Waiting for submitted prediction job")
            time.sleep(seconds_to_sleep)
            seconds_to_sleep = seconds_between_polls
            if isinstance(job, _DummyJobIterable):
                # dummy job runs in the same thread, so we need to call result to let it run
                _ = job.result(timeout_seconds)

            outs: Any = job.outputs()  # pyright: ignore[reportUnknownVariableType]
            assert is_sequence(outs)
            outs = outs[output_idx:]
            for out in outs:
                yield out
                output_idx += 1

    # cannot json serialize any iterable, so we need to convert to a list first
    # this also makes it trivial to not exhaust the iterable in case we need to retry with model upload
    serialized_input_sample_list = list(serialized_input_sample)

    try_with_model_upload = True
    try:
        job = submit("", input_sample=serialized_input_sample_list)
        for block in iterate_job(job):
            yield block
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
            # An exception occured somehow after successfully obtaining the first block of the result
            # even though the server should only return a single block as we only submitted a single block.
            raise

    if try_with_model_upload:
        job = submit(
            serialized_model,
            input_sample=serialized_input_sample_list,
        )
        for block in iterate_job(job):
            yield block
