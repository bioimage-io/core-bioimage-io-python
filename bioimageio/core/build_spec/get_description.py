import hashlib
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import FilePath

# from bioimageio.core import export_resource_package, load_raw_resource_description
from typing_extensions import NotRequired, Self, Unpack

from bioimageio.core.io import FileSource, download, load_description_and_validate, write_description
from bioimageio.core.utils import get_sha256
from bioimageio.spec.description import ValidationContext
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


class _CoreGenericBaseKwargs(TypedDict):
    name: str
    description: str
    authors: NotEmpty[Sequence[Author]]
    maintainers: NotRequired[Sequence[Maintainer]]
    tags: Sequence[str]
    documentation: FileSource
    cite: NotEmpty[Sequence[CiteEntry]]
    license: LicenseId
    output_path: Path


class _TensorKwargs(TypedDict):
    test_tensor: FileSource
    sample_tensor: NotRequired[FileSource]
    id: NotRequired[Optional[TensorId]]
    data: NotRequired[Optional[Union[TensorData, NotEmpty[Sequence[TensorData]]]]]
    output_path: Path


class _OutputTensorKwargs(_TensorKwargs):
    axes: NotEmpty[Sequence[OutputAxis]]
    postprocessing: NotRequired[Sequence[Postprocessing]]


class SpecBuilder:
    def __init__(self, output_path: Path, output_path_exist_ok: bool = False) -> None:
        super().__init__()
        output_path.mkdir(parents=True, exist_ok=output_path_exist_ok)
        self.output_path = output_path

    def include_file(self, source: FileSource) -> RelativeFilePath:
        local_source = download(source)
        try:
            rel_path = local_source.path.relative_to(
                self.output_path
            )  # todo: improve for py >= 3.9 with path.is_relative_to
        except ValueError:
            # local source is not in output path
            dest_path = self.output_path / local_source.original_file_name
            if dest_path.exists():
                file_hash = get_sha256(local_source.path)
                for i in range(10):
                    dest_hash = get_sha256(dest_path)
                    if dest_hash == file_hash:
                        break

                    dest_path = dest_path.with_name(f"{dest_path.stem}-{i}{dest_path.suffix}")
                    if not dest_path.exists():
                        break
                else:
                    raise RuntimeError("Encountered too many unidentical files with the same file name.")

            if not dest_path.exists():
                _ = Path(shutil.copy(local_source.path, dest_path))

            rel_path = dest_path.relative_to(self.output_path)

        return RelativeFilePath(rel_path)


class ModelBuilder(SpecBuilder):
    def add_cite(self):
        self._cite.append(CiteEntry())

    def add_input_tensor(
        self,
        *,
        test_tensor: Union[NDArray[Any], FileSource],
        axes: Sequence[InputAxis],
        preprocessing: Sequence[Preprocessing],
        id_: TensorId,
        data: TensorData,
        sample_tensor: Optional[FileSource],
    ) -> InputTensor:
        return InputTensor.model_validate(
            InputTensor(
                test_tensor=self.include_file(test_tensor),
                id=id_,
                axes=tuple(axes),
                preprocessing=tuple(preprocessing),
                data=data,
                sample_tensor=None if sample_tensor is None else self.include_file(sample_tensor),
            ),
            context=self.context,
        )

    # def add_input_tensor()
    # def add_cover_image(cover)
    def build(self, output_path: Path, *, inputs: Sequence[InputTensor]):
        assert False


mb = ModelBuilder(Path("output_path"))
mb.build(inputs=[mb.build_input_tensor(test_tensor=tt) for tt in test_tensors], outputs=based_on.outputs)


class SpecGuesser:
    @staticmethod
    def guess_data_range(array: NDArray[Any]):
        if np.issubdtype(array.dtype, np.floating) and array.min() >= 0.0 and array.max() <= 1.0:
            return (0.0, 1.0)
        else:
            return (None, None)

    @classmethod
    def guess_data_description(cls, test_tensor: FileSource):
        try:
            array: Union[Any, NDArray[Any]] = np.load(download(test_tensor).path)
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Expected numpy array, but got {type(array)}")
        except Exception as e:
            warnings.warn(f"Could not guess data type of {test_tensor}: {e}")
            return None

        dtype_str = str(array.dtype)
        dtype: IntervalOrRatioDType = dtype_str  # type: ignore  # validated by IntervalOrRatioData
        return IntervalOrRatioData(
            type=dtype, range=cls.guess_data_range(array), unit="arbitrary unit", scale=1.0, offset=None
        )


class SpecBuilderWithGuesses(SpecBuilder, SpecGuesser):
    # def __init__(self, output_path: Path) -> None:
    #     super().__init__(output_path)

    def build_input_tensor(
        self,
        *,
        test_tensor: FileSource,
        axes: Sequence[InputAxis],
        preprocessing: Sequence[Preprocessing],
        id_: TensorId,
        data: Optional[TensorData] = None,
        sample_tensor: FileSource | None,
    ) -> InputTensor:
        return super().build_input_tensor(
            test_tensor=test_tensor,
            axes=axes,
            preprocessing=preprocessing,
            id_=id_,
            data=data or self.guess_data_description(test_tensor),
            sample_tensor=sample_tensor,
        )


def build_spec_interactively(output_path: Path):
    guesser = SpecGuesser(output_path)
    builder = SpecBuilder(output_path)


class _CoreOutputTensor(OutputTensor, _CoreTensorMixin):
    @classmethod
    def build(cls, **kwargs: Unpack[_OutputTensorKwargs]):
        return cls(
            test_tensor=ensure_file_in_folder(kwargs["test_tensor"], kwargs["output_path"]),
            id=kwargs.get("id") or TensorId("output"),
            axes=tuple(kwargs["axes"]),
            postprocessing=tuple(kwargs.get("postprocessing", ())),
            data=cls.get_data_description(kwargs),
        )


class CoreModelBaseKwargs(_CoreGenericBaseKwargs):
    inputs: NotEmpty[Sequence[_InputTensorKwargs]]
    outputs: NotEmpty[Sequence[_OutputTensorKwargs]]


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
    loaded = load_description_and_validate(descr_path)
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
