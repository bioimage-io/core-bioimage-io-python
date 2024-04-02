import collections.abc
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from numpy.typing import NDArray
from typing_extensions import assert_never

from bioimageio.spec.model import AnyModelDescr, v0_4
from bioimageio.spec.model.v0_5 import WeightsFormat

from .axis import AxisInfo
from .model_adapters import ModelAdapter, create_model_adapter
from .model_adapters import get_weight_formats as get_weight_formats
from .proc_ops import Processing
from .proc_setup import setup_pre_and_postprocessing
from .sample import TiledSample, UntiledSample
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor, TensorId
from .tile import Tile
from .utils import get_axes_infos


@dataclass
class CoreTensorDescr:
    id: TensorId
    axes: Sequence[AxisInfo]
    optional: bool


Data = TypeVar(
    "Data",
    TiledSample,
    UntiledSample,
    Tile,
    Iterable[TiledSample],
    Iterable[UntiledSample],
    NDArray[Any],
    Sequence[Optional[NDArray[Any]]],
    Mapping[Union[TensorId, str], Optional[NDArray[Any]]],
)


class PredictionPipeline:
    """
    Represents model computation including preprocessing and postprocessing
    Note: Ideally use the PredictionPipeline as a context manager
    """

    def __init__(
        self,
        *,
        name: str,
        bioimageio_model: AnyModelDescr,
        preprocessing: List[Processing],
        postprocessing: List[Processing],
        model: ModelAdapter,
    ) -> None:
        super().__init__()
        if bioimageio_model.run_mode:
            warnings.warn(
                f"Not yet implemented inference for run mode '{bioimageio_model.run_mode.name}'"
            )

        self.name = name
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self.input_ids = tuple(
            (TensorId(str(t.name)) if isinstance(t, v0_4.InputTensorDescr) else t.id)
            for t in bioimageio_model.inputs
        )
        self.inputs = collections.OrderedDict(
            (
                tid,
                CoreTensorDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=not isinstance(t, v0_4.InputTensorDescr) and t.optional,
                ),
            )
            for tid, t in zip(self.input_ids, bioimageio_model.inputs)
        )
        self.output_ids = tuple(
            (TensorId(str(t.name)) if isinstance(t, v0_4.OutputTensorDescr) else t.id)
            for t in bioimageio_model.outputs
        )
        self.outputs = collections.OrderedDict(
            (
                tid,
                CoreTensorDescr(
                    id=tid,
                    axes=get_axes_infos(t),
                    optional=False,
                ),
            )
            for tid, t in zip(self.output_ids, bioimageio_model.outputs)
        )

        self._adapter: ModelAdapter = model

    def __call__(self, data: Data) -> Data:
        return self.predict(data)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.unload()
        return False

    def predict(self, inputs: Data) -> Data:
        """Run model prediction **including** pre/postprocessing."""

        if isinstance(inputs, Tile):
            self.apply_preprocessing(inputs)
            output_tile = Tile(
                data={
                    tid: out
                    for tid, out in zip(
                        self.output_ids,
                        self._adapter.forward(
                            *(inputs.data[t] for t in self.input_ids)
                        ),
                    )
                    if out is not None
                }
            )
            self.apply_postprocessing(output_tile)
            return output_tile

        else:
            assert_never(inputs)

        return output

        # if isinstance(inputs, collections.abc.Mapping):
        #     data = {
        #         tid: d
        #         for tid in self.input_ids
        #         if (d := inputs.get(tid, inputs.get(str(tid)))) is not None
        #     }
        # else:
        #     if isinstance(inputs, (Tensor, np.ndarray)):
        #         inputs_seq = [inputs]
        #     else:
        #         inputs_seq = inputs

        #     assert len(inputs_seq) == len(self.input_ids)
        #     data = {
        #         tid: d for tid, d in zip(self.input_ids, inputs_seq) if d is not None
        #     }

        # sample = UntiledSample(
        #     data={
        #         tid: Tensor.from_numpy(d, dims=self.inputs[tid].axes)
        #         for tid, d in data.items()
        #     }
        # )
        # output = self.predict(sample)
        # return {tid: out.data.data for }

    def apply_preprocessing(self, tile: Tile) -> None:
        """apply preprocessing in-place, also updates sample stats"""
        for op in self._preprocessing:
            op(tile)

    def apply_postprocessing(self, tile: Tile) -> None:
        """apply postprocessing in-place, also updates samples stats"""
        for op in self._postprocessing:
            op(tile)

    def load(self):
        """
        optional step: load model onto devices before calling forward if not using it as context manager
        """
        pass

    def unload(self):
        """
        free any device memory in use
        """
        self._adapter.unload()


def create_prediction_pipeline(
    bioimageio_model: AnyModelDescr,
    *,
    devices: Optional[Sequence[str]] = None,
    weight_format: Optional[WeightsFormat] = None,
    weights_format: Optional[WeightsFormat] = None,
    dataset_for_initial_statistics: Iterable[
        Union[UntiledSample, Sequence[Tensor]]
    ] = tuple(),
    keep_updating_initial_dataset_statistics: bool = False,
    fixed_dataset_statistics: Mapping[DatasetMeasure, MeasureValue] = MappingProxyType(
        {}
    ),
    model_adapter: Optional[ModelAdapter] = None,
    **deprecated_kwargs: Any,
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * computation of input statistics
    * preprocessing
    * model prediction
    * computation of output statistics
    * postprocessing
    """
    weights_format = weight_format or weights_format
    del weight_format
    if deprecated_kwargs:
        warnings.warn(
            f"deprecated create_prediction_pipeline kwargs: {set(deprecated_kwargs)}"
        )

    model_adapter = model_adapter or create_model_adapter(
        model_description=bioimageio_model,
        devices=devices,
        weight_format_priority_order=weights_format and (weights_format,),
    )

    if isinstance(bioimageio_model, v0_4.ModelDescr):
        input_ids = [TensorId(str(ipt.name)) for ipt in bioimageio_model.inputs]
    else:
        input_ids = [ipt.id for ipt in bioimageio_model.inputs]

    def dataset():
        for x in dataset_for_initial_statistics:
            if isinstance(x, UntiledSample):
                yield x
            else:
                yield UntiledSample(data=dict(zip(input_ids, x)))

    preprocessing, postprocessing = setup_pre_and_postprocessing(
        bioimageio_model,
        dataset(),
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_statistics,
        fixed_dataset_stats=fixed_dataset_statistics,
    )

    return PredictionPipeline(
        name=bioimageio_model.name,
        bioimageio_model=bioimageio_model,
        model=model_adapter,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
    )
