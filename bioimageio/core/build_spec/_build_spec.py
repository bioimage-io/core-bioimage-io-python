import collections.abc
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Type, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

# from bioimageio.core import export_resource_package, load_raw_resource_description
from typing_extensions import NotRequired, Self, Unpack

from bioimageio.core.io import FileSource, download, read_description_and_validate, write_description
from bioimageio.spec.model.v0_5 import (
    Architecture,
    Author,
    CiteEntry,
    Dependencies,
    InputAxis,
    InputTensor,
    IntervalOrRatioData,
    IntervalOrRatioDType,
    LicenseId,
    Maintainer,
    Model,
    NotEmpty,
    OutputAxis,
    OutputTensor,
    Postprocessing,
    Preprocessing,
    PytorchStateDictWeights,
    RelativeFilePath,
    Sha256,
    TensorData,
    TensorId,
    Version,
    Weights,
)


class CoreGenericBaseKwargs(TypedDict):
    name: str
    description: str
    authors: NotEmpty[Sequence[Author]]
    maintainers: NotRequired[Sequence[Maintainer]]
    tags: Sequence[str]
    documentation: FileSource
    cite: NotEmpty[Sequence[CiteEntry]]
    license: LicenseId
    output_path: Path


class CoreTensorKwargs(TypedDict):
    test_tensor: FileSource
    sample_tensor: NotRequired[FileSource]
    id: NotRequired[Optional[TensorId]]
    data: NotRequired[Optional[Union[TensorData, NotEmpty[Sequence[TensorData]]]]]
    output_path: Path


class CoreInputTensorKwargs(CoreTensorKwargs):
    axes: NotEmpty[Sequence[InputAxis]]
    preprocessing: NotRequired[Sequence[Preprocessing]]


class CoreOutputTensorKwargs(CoreTensorKwargs):
    axes: NotEmpty[Sequence[OutputAxis]]
    postprocessing: NotRequired[Sequence[Postprocessing]]


def ensure_file_in_folder(source: FileSource, folder: Path) -> RelativeFilePath:
    """download/copy `source` to `folder` if `source` is not already in (a subfolder of) `folder`.
    Returns a relative file path (relative to `folder`)"""
    path = download(source).path
    try:
        rel_path = path.relative_to(folder)  # todo: improve for py >= 3.9 with path.is_relative_to
    except ValueError:
        path = Path(shutil.copy(path, folder))
        rel_path = path.relative_to(folder)

    return RelativeFilePath(rel_path)


class _CoreTensorMixin:
    @staticmethod
    def get_data_description(kwargs: Union[CoreInputTensorKwargs, CoreOutputTensorKwargs]):
        tensor_data = kwargs.get("data")
        if isinstance(tensor_data, TensorData):
            return tensor_data
        elif tensor_data is None:
            test_tensor: NDArray[Any] = np.load(download(kwargs["test_tensor"]).path)
            assert isinstance(test_tensor, np.ndarray)
            dtype_str = str(test_tensor.dtype)
            if dtype_str.startswith("float") and test_tensor.min() >= 0.0 and test_tensor.max() <= 1.0:
                range_ = (0.0, 1.0)
            else:
                range_ = (None, None)

            dtype: IntervalOrRatioDType = dtype_str  # type: ignore  # validated by IntervalOrRatioData
            return IntervalOrRatioData(type=dtype, range=range_, unit="arbitrary unit", scale=1.0, offset=None)
        elif isinstance(tensor_data, collections.abc.Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
            assert all(isinstance(td, TensorData) for td in tensor_data)
            return tuple(tensor_data)
        else:
            raise TypeError(tensor_data)


class _CoreInputTensor(InputTensor, _CoreTensorMixin):
    @classmethod
    def build(cls, **kwargs: Unpack[CoreInputTensorKwargs]):
        return cls(
            test_tensor=ensure_file_in_folder(kwargs["test_tensor"], kwargs["output_path"]),
            id=kwargs.get("id") or TensorId("input"),
            axes=tuple(kwargs["axes"]),
            preprocessing=tuple(kwargs.get("preprocessing", ())),
            data=cls.get_data_description(kwargs),
            sample_tensor=ensure_file_in_folder(kwargs["sample_tensor"], kwargs["output_path"])
            if "sample_tensor" in kwargs
            else None,
        )


class _CoreOutputTensor(OutputTensor, _CoreTensorMixin):
    @classmethod
    def build(cls, **kwargs: Unpack[CoreOutputTensorKwargs]):
        return cls(
            test_tensor=ensure_file_in_folder(kwargs["test_tensor"], kwargs["output_path"]),
            id=kwargs.get("id") or TensorId("output"),
            axes=tuple(kwargs["axes"]),
            postprocessing=tuple(kwargs.get("postprocessing", ())),
            data=cls.get_data_description(kwargs),
        )


class CoreModelBaseKwargs(CoreGenericBaseKwargs):
    inputs: NotEmpty[Sequence[CoreInputTensorKwargs]]
    outputs: NotEmpty[Sequence[CoreOutputTensorKwargs]]


class CoreModelKwargs(CoreModelBaseKwargs):
    weights: Weights


class _CoreModel(Model):
    @classmethod
    def build(cls, **kwargs: Unpack[CoreModelKwargs]) -> Self:
        documentation = ensure_file_in_folder(kwargs["documentation"], kwargs["output_path"])

        inputs = tuple(
            _CoreInputTensor.build(
                id=t_kwargs["id"] if "id" in t_kwargs else TensorId(f"input{i}"),
                test_tensor=t_kwargs["test_tensor"],
                axes=t_kwargs["axes"],
                data=t_kwargs.get("data"),
                output_path=kwargs["output_path"],
            )
            for i, t_kwargs in enumerate(kwargs["inputs"])
        )

        outputs = tuple(
            _CoreOutputTensor.build(
                id=t_kwargs["id"] if "id" in t_kwargs else TensorId(f"output{i}"),
                test_tensor=t_kwargs["test_tensor"],
                axes=t_kwargs["axes"],
                data=t_kwargs.get("data"),
                output_path=kwargs["output_path"],
            )
            for i, t_kwargs in enumerate(kwargs["outputs"])
        )

        return cls(
            name=kwargs["name"],
            description=kwargs["description"],
            authors=tuple(kwargs["authors"]),
            maintainers=tuple(kwargs.get("maintainers", ())),
            cite=tuple(kwargs["cite"]),
            license=kwargs["license"],
            timestamp=datetime.now(),
            inputs=inputs,
            outputs=outputs,
            weights=kwargs["weights"],
            documentation=documentation,
        )

    @classmethod
    def build_from_pytorch_state_dict(
        cls,
        weights: FileSource,
        architecture: Architecture,
        sha256: Optional[Sha256] = None,
        pytorch_version: Optional[Version] = None,
        dependencies: Optional[Dependencies] = None,
        **kwargs: Unpack[CoreModelBaseKwargs],
    ):
        if pytorch_version is None:
            import torch

            pytorch_version = Version(torch.__version__)

        return cls.build(
            weights=Weights(
                pytorch_state_dict=PytorchStateDictWeights(
                    source=ensure_file_in_folder(weights, kwargs["output_path"]),
                    sha256=sha256,
                    architecture=architecture,
                    pytorch_version=pytorch_version,
                    dependencies=dependencies,
                )
            ),
            **kwargs,
        )


def _build_spec_common(core_descr: _CoreModel, descr_path: Path, expected_type: Type[Any]):
    write_description(core_descr, descr_path)
    loaded = read_description_and_validate(descr_path)
    if type(loaded) is not expected_type:
        raise RuntimeError(f"Created {descr_path} was loaded as {type(loaded)}, but expected {expected_type}")

    return descr_path, loaded


def build_model_spec(
    *,
    weights: FileSource,
    architecture: Architecture,
    sha256: Optional[Sha256] = None,
    pytorch_version: Optional[Version] = None,
    dependencies: Optional[Dependencies] = None,
    **kwargs: Unpack[CoreModelBaseKwargs],
):
    model = _CoreModel.build_from_pytorch_state_dict(
        weights=weights,
        architecture=architecture,
        sha256=sha256,
        pytorch_version=pytorch_version,
        dependencies=dependencies,
        **kwargs,
    )

    return _build_spec_common(model, kwargs["output_path"] / "description.bioimageio.yaml", Model)
