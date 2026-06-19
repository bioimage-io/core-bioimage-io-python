import gc
import warnings
from abc import ABC, abstractmethod
from queue import LifoQueue
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Union

from exceptiongroup import ExceptionGroup
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import TypeVar

from bioimageio.spec import ValidationSummary
from bioimageio.spec.model import AnyModelDescr, v0_4

from ._sample_serializer import SampleSerializer, SerializedSampleBlockType
from .common import PerMember
from .digest_spec import get_axes_infos, get_member_ids
from .sample import Sample
from .tensor import Tensor


class ModelAdapter(ABC):
    """
    Represents model *without* any preprocessing or postprocessing.

    ```
    from bioimageio.core import load_description

    model = load_description(...)

    # option 1:
    adapter = create_model_adapter(model)
    adapter.forward(...)
    adapter.unload()

    # option 2:
    with create_model_adapter(model) as adapter:
        adapter.forward(...)
    ```
    """

    def __init__(
        self, model_description: AnyModelDescr, devices: Optional[Sequence[str]]
    ):
        super().__init__()
        self._model_descr = model_description
        self._input_ids = get_member_ids(model_description.inputs)
        self._output_ids = get_member_ids(model_description.outputs)
        self._input_axes = [
            tuple(a.id for a in get_axes_infos(t)) for t in model_description.inputs
        ]
        self._output_axes = [
            tuple(a.id for a in get_axes_infos(t)) for t in model_description.outputs
        ]
        if isinstance(model_description, v0_4.ModelDescr):
            self._input_is_optional = [False] * len(model_description.inputs)
        else:
            self._input_is_optional = [ipt.optional for ipt in model_description.inputs]

        self._devices = devices
        self.load()

    @property
    def model_descr(self) -> AnyModelDescr:
        return self._model_descr

    @abstractmethod
    def load(self) -> None:
        self._loaded = True

    @abstractmethod
    def forward(
        self, inputs: PerMember[Optional[Tensor]]
    ) -> PerMember[Optional[Tensor]]: ...

    @abstractmethod
    def unload(self):
        """Unload model from any devices, freeing their memory.

        Note:
            The moder adapter should be considered unusable afterwards.
        """
        self._loaded = False

    def close(self):
        """Close the model adapter, freeing any resources.

        Note:
            The moder adapter should be considered unusable afterwards.
        """
        self.unload()


DeviceType = TypeVar("DeviceType")
ModelType = TypeVar("ModelType")


class LocalModelAdapter(ModelAdapter, ABC, Generic[DeviceType, ModelType]):
    def load(self) -> None:
        devices = self._devices
        self._model_queue: LifoQueue[Tuple[DeviceType, ModelType]] = LifoQueue()
        parsed_devices = self._parse_devices(devices)
        assert parsed_devices
        # prioritize devices by order specified by user
        device_exceptions: Dict[str, Exception] = {}
        self._initialized_devices: List[str] = []
        for d in parsed_devices[::-1]:
            try:
                model = self._init_model_on_device(d)
            except Exception as e:
                device_exceptions[str(d)] = e
            else:
                self._model_queue.put((d, model))
                self._initialized_devices.insert(0, str(d))

        if self._model_queue.empty():
            if len(device_exceptions) == 1:
                raise next(iter(device_exceptions.values()))
            else:
                raise ExceptionGroup(
                    "Failed to initialize model on any of the requested devices.",
                    list(device_exceptions.values())[::-1],
                )

        if device_exceptions:
            logger.warning(
                "Failed to initialize model on some of the requested devices. Successfully initialized on {}, but got the following errors for other devices: {}",
                self._initialized_devices,
                device_exceptions,
            )

        super().load()

    @abstractmethod
    def _parse_devices(self, devices: Optional[Sequence[str]]) -> Sequence[DeviceType]:
        """Parse devices

        Note:
            - May not return an empty sequence.
            - The order of devices in the returned sequence determines the priority of device usage in the forward pass.
              First devices has highgest priority, last device has lowest priority.
        """

    @abstractmethod
    def _init_model_on_device(self, device: DeviceType) -> ModelType: ...

    def forward(
        self, inputs: PerMember[Optional[Tensor]]
    ) -> PerMember[Optional[Tensor]]:
        """
        Run forward pass of model to get model predictions

        Note: sample id and stample stat attributes are passed through
        """
        if not self._loaded:
            raise RuntimeError("Model must be `.load()`ed before calling forward()")

        unexpected = [mid for mid in inputs if mid not in self._input_ids]
        if unexpected:
            warnings.warn(f"Got unexpected input tensor IDs: {unexpected}")

        input_arrays = [
            (
                None
                if (a := inputs.get(in_id)) is None
                else a.transpose(in_order).data.data
            )
            for in_id, in_order in zip(self._input_ids, self._input_axes)
        ]
        logger.debug(
            "NN input shapes: {}",
            [a.shape if a is not None else None for a in input_arrays],
        )
        device, model = self._model_queue.get()
        try:
            output_arrays = self._forward_impl(device, model, input_arrays)
        finally:
            self._model_queue.put((device, model))

        logger.debug(
            "NN output shapes: {}",
            [a.shape if a is not None else None for a in output_arrays],
        )
        if len(output_arrays) > len(self._output_ids):
            warnings.warn(
                f"Model produced more outputs ({len(output_arrays)}) than specified in the model description ({len(self._output_ids)}). Extra outputs will be ignored."
            )
            output_arrays = output_arrays[: len(self._output_ids)]

        output_tensors = [
            None if a is None else Tensor(a, dims=d)
            for a, d in zip(output_arrays, self._output_axes)
        ]
        return {
            tid: out
            for tid, out in zip(
                self._output_ids,
                output_tensors,
            )
            if out is not None
        }

    @abstractmethod
    def _forward_impl(
        self,
        device: DeviceType,
        model: ModelType,
        input_arrays: Sequence[Optional[NDArray[Any]]],
    ) -> Union[List[Optional[NDArray[Any]]], Tuple[Optional[NDArray[Any]], ...]]:
        """framework specific forward implementation"""

    def unload(self):
        for _ in range(len(self._initialized_devices)):
            device, model = self._model_queue.get()
            try:
                self._cleanup_pre_model_deletion(device, model)
            except Exception as e:
                logger.warning(
                    "Got error during pre-deletion cleanup on device {}: {}", device, e
                )
            finally:
                del model
            try:
                self._cleanup_post_model_deletion(device)
            except Exception as e:
                logger.warning(
                    "Got error during post-deletion cleanup on device {}: {}", device, e
                )

        _ = gc.collect()  # deallocate memory
        super().unload()

    @abstractmethod
    def _cleanup_pre_model_deletion(self, device: DeviceType, model: ModelType) -> None:
        """Clean up before model reference deletion"""

    @abstractmethod
    def _cleanup_post_model_deletion(self, device: DeviceType) -> None:
        """Clean up after model reference deletion"""


class RemoteModelAdapter(ModelAdapter, ABC, Generic[SerializedSampleBlockType]):
    """Model adapter to use a remote service for model inference."""

    def __init__(
        self,
        model_description: AnyModelDescr,
        server: str,
        sample_serializer: SampleSerializer[SerializedSampleBlockType],
    ):
        super().__init__(model_description, devices=None)
        self._server = server
        self._serializer = sample_serializer

    @property
    def server(self) -> str:
        return self._server

    def forward(
        self, inputs: PerMember[Optional[Tensor]]
    ) -> PerMember[Optional[Tensor]]:
        serialized_input = self._serializer.serialize_sample(
            Sample(
                members={k: v for k, v in inputs.items() if v is not None},
                stat={},
                id=None,
            )
        )
        serialized_output = self._forward_impl(serialized_input)
        return self._serializer.deserialize_sample(serialized_output).members

    @abstractmethod
    def _forward_impl(
        self, serialized_input_sample: Iterable[SerializedSampleBlockType]
    ) -> Iterable[SerializedSampleBlockType]: ...

    @abstractmethod
    def test(self) -> Optional[ValidationSummary]:
        """Run the bioimageio model test."""
