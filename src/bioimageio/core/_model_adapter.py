from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from queue import LifoQueue
from typing import (
    Any,
    Generic,
    Literal,
    cast,
)

import dask.array as da
from dask.delayed import Delayed, delayed
from exceptiongroup import ExceptionGroup
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import TypeVar, assert_never

from bioimageio.spec import ValidationSummary
from bioimageio.spec.model import AnyModelDescr, v0_4

from ._sample_serializer import SampleSerializer, SerializedSampleBlockType
from ._utils.type_hints import NestedList, cast_and_assert
from .axis import AxisId, PerAxis
from .common import PerMember
from .digest_spec import (
    get_axes_infos,
    get_member_ids,
    get_unstacked_output_dim_orders_for_model,
)
from .sample import Sample
from .tensor import Array, MultiIndex, Tensor, restore_batch_multi_index


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
        self,
        model_description: AnyModelDescr,
        devices: Sequence[str] | None,
        computation_mode: Literal["eager", "lazy"] = "lazy",
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
            if computation_mode == "lazy":
                warnings.warn(
                    "'lazy' computation mode is not supported for v0.4 model descriptions. Falling back to 'eager' computation mode."
                )
                computation_mode = "eager"

            self._input_is_optional = [False] * len(model_description.inputs)
        else:
            self._input_is_optional = [ipt.optional for ipt in model_description.inputs]

        self.computation_mode: Literal["eager", "lazy"] = computation_mode

        self._devices = devices
        self._loaded = False

    def get_output_shape(
        self,
        input_shape: PerMember[PerAxis[int]],
    ) -> PerMember[PerAxis[int]]:
        if isinstance(self._model_descr, v0_4.ModelDescr):
            raise NotImplementedError(
                "v0_4.ModelDescr.get_output_shape is not implemented. Please update the model description to v0_5 or later."
            )
        return {
            m: {d: s if isinstance(s, int) else -1 for d, s in s_per_a.items()}
            for m, s_per_a in self._model_descr.get_output_tensor_sizes(
                input_shape
            ).items()
        }

    @property
    def model_descr(self) -> AnyModelDescr:
        return self._model_descr

    @property
    def is_loaded(self) -> bool:
        """`True` if the model adapter is loaded and ready for inference."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model adapter and prepare it for inference."""
        self._loaded = True

    @abstractmethod
    def forward(
        self, inputs: PerMember[Tensor | Array | None]
    ) -> PerMember[Tensor | None]:
        """
        Run forward pass of model to get model predictions

        Args:
            inputs: input tensors for the model, keyed by member id
        """

    @abstractmethod
    def unload(self):
        """Unload model from any devices, freeing their memory.

        Note:
            The model adapter should be considered unusable afterwards.
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
        self._model_queue: LifoQueue[tuple[DeviceType, ModelType]] = LifoQueue()
        parsed_devices = self._parse_devices(devices)
        assert parsed_devices
        # prioritize devices by order specified by user
        device_exceptions: dict[str, Exception] = {}
        self._initialized_devices: list[str] = []
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
    def _parse_devices(self, devices: Sequence[str] | None) -> Sequence[DeviceType]:
        """Parse devices

        Note:
            - May not return an empty sequence.
            - The order of devices in the returned sequence determines the priority of device usage in the forward pass.
              First devices has highgest priority, last device has lowest priority.
        """

    @abstractmethod
    def _init_model_on_device(self, device: DeviceType) -> ModelType: ...

    def forward(
        self, inputs: PerMember[Tensor | Array | None]
    ) -> PerMember[Tensor | None]:
        """
        Run forward pass of model to get model predictions

        Args:
            inputs: input tensors for the model, keyed by member id
        """

        if not self._loaded:
            raise RuntimeError("Model must be loaded before calling forward()")

        unexpected = [mid for mid in inputs if mid not in self._input_ids]
        if unexpected:
            warnings.warn(f"Got unexpected input tensor IDs: {unexpected}")

        multi_batch_indices: dict[str, MultiIndex] = {}
        input_arrays: list[Array | None] = []
        for in_id, in_order in zip(self._input_ids, self._input_axes):
            if isinstance((a := inputs.get(in_id)), Tensor):
                a_transposed = a.transpose(
                    in_order,
                    extra_dims="stack" if a.batch_multi_index is None else "raise",
                )
                if a_transposed.batch_multi_index is not None:
                    multi_batch_indices[in_id] = a_transposed.batch_multi_index

                a = a_transposed.data.data

            input_arrays.append(a)

        logger.debug(
            "NN input shapes: {}",
            [a.shape if a is not None else None for a in input_arrays],
        )

        if self.computation_mode == "eager":
            output_arrays = self._forward_arrays(*input_arrays)
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
            outputs = {
                tid: out
                for tid, out in zip(
                    self._output_ids,
                    output_tensors,
                )
                if out is not None
            }
        elif self.computation_mode == "lazy":
            deferred_inputs = [
                cast_and_assert(
                    Delayed,
                    a.rechunk(
                        a.shape  # pyright: ignore[reportArgumentType]
                    )
                    .to_delayed(optimize_graph=False)
                    .item(),
                )
                if isinstance(a, da.Array)
                else a
                for a in input_arrays
            ]

            output_arrays = cast(
                Sequence[Delayed | None],
                delayed(
                    self._forward_arrays,
                    pure=False,
                    nout=len(self._output_ids),
                )(*deferred_inputs),
            )

            in_shape = {
                m: {d: int(s) for d, s in zip(dims, a.shape)}
                for m, a, dims in zip(self._input_ids, input_arrays, self._input_axes)
                if a is not None
            }
            out_shape = self.get_output_shape(in_shape)
            outputs = {
                m: Tensor(
                    da.from_delayed(
                        a, shape=[out_shape[m][d] for d in dims], dtype=dtype
                    ),
                    dims=dims,
                )
                for m, a, dims, dtype in zip(
                    self._output_ids,
                    output_arrays,
                    self._output_axes,
                    [out.dtype for out in self._model_descr.outputs],
                )
                if a is not None
            }
        else:
            assert_never(self.computation_mode)

        if multi_batch_indices:
            outputs = restore_batch_multi_index(multi_batch_indices.values(), outputs)
            # Reorder unstacked output dims to match the order of the unstacked input dims, if possible
            unstacked_output_dim_orders = get_unstacked_output_dim_orders_for_model(
                self._model_descr,
                {
                    m: in_tensor.dims
                    for m, in_tensor in inputs.items()
                    if isinstance(in_tensor, Tensor)
                },
                stacked_output_dims={
                    m: [
                        AxisId(name)
                        for name in out.batch_multi_index.names
                        if name is not None
                    ]
                    for m, out in outputs.items()
                    if out.batch_multi_index is not None
                },
            )
            outputs = {
                m: out.transpose(unstacked_output_dim_orders[m])
                for m, out in outputs.items()
            }

        return outputs

    def _forward_arrays(self, *input_arrays: Array | None):
        if not self._loaded:
            raise RuntimeError(
                "Cannot compute model outputs because the model adapter is unloaded. "
                + "Compute lazy outputs inside the prediction pipeline context "
                + "(`with pred_pipe as pp:`), before the model is unloaded."
            )
        device, model = self._model_queue.get()
        try:
            return self._forward_impl(
                device,
                model,
                [
                    cast(NDArray[Any], a.compute()) if isinstance(a, da.Array) else a
                    for a in input_arrays
                ],
            )
        finally:
            self._model_queue.put((device, model))

    @abstractmethod
    def _forward_impl(
        self,
        device: DeviceType,
        model: ModelType,
        input_arrays: Sequence[NDArray[Any] | None],
    ) -> list[NDArray[Any] | None] | tuple[NDArray[Any] | None, ...]:
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
        self, inputs: PerMember[Tensor | Array | None]
    ) -> PerMember[Tensor | None]:
        input_dims = dict(zip(self._input_ids, self._input_axes))
        serialized_input = self._serializer.serialize_sample(
            Sample(
                members={
                    k: v if isinstance(v, Tensor) else Tensor(v, dims=input_dims[k])
                    for k, v in inputs.items()
                    if v is not None
                },
                stat={},
                id=None,
            )
        )
        serialized_output = self._forward_impl(serialized_input)
        output = self._serializer.deserialize_sample(serialized_output).members
        return output

    @abstractmethod
    def _forward_impl(
        self, serialized_input_sample: NestedList[SerializedSampleBlockType]
    ) -> NestedList[SerializedSampleBlockType]: ...

    @abstractmethod
    def test(self) -> ValidationSummary | None:
        """Run the bioimageio model test."""
