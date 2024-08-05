from functools import cached_property
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from loguru import logger
from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from tqdm import tqdm

from bioimageio.core import (
    MemberId,
    Sample,
    __version__,
    create_prediction_pipeline,
)
from bioimageio.core.commands import WeightFormatArg, package, test, validate_format
from bioimageio.core.common import SampleId
from bioimageio.core.digest_spec import load_sample_for_model
from bioimageio.core.io import save_sample
from bioimageio.core.proc_setup import (
    DatasetMeasure,
    Measure,
    MeasureValue,
    StatsCalculator,
    get_required_dataset_measures,
)
from bioimageio.core.stat_measures import Stat
from bioimageio.spec import (
    AnyModelDescr,
    InvalidDescr,
    load_description,
)
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr, v0_4, v0_5
from bioimageio.spec.notebook import NotebookDescr
from bioimageio.spec.utils import ensure_description_is_model


class CmdBase(BaseModel, use_attribute_docstrings=True):
    pass


class ArgMixin(BaseModel, use_attribute_docstrings=True):
    pass


class WithSource(ArgMixin):
    source: CliPositionalArg[str]
    """Url/path to a bioimageio.yaml/rdf.yaml file or a bioimage.io resource identifier, e.g. 'affable-shark'"""

    @cached_property
    def descr(self):
        return load_description(self.source, perform_io_checks=False)

    @property
    def descr_id(self) -> str:
        """a more user-friendly description id
        (replacing legacy ids with their nicknames)
        """
        if isinstance(self.descr, InvalidDescr):
            return str(getattr(self.descr, "id", getattr(self.descr, "name")))
        else:
            return str(
                (
                    (bio_config := self.descr.config.get("bioimageio", {}))
                    and isinstance(bio_config, dict)
                    and bio_config.get("nickname")
                )
                or self.descr.id
                or self.descr.name
            )


class ValidateFormatCmd(CmdBase, WithSource):
    """bioimageio-validate-format - validate the meta data format of a bioimageio resource."""

    def run(self):
        validate_format(self.descr)


class TestCmd(CmdBase, WithSource):
    """bioimageio-test - Test a bioimageio resource (beyond meta data formatting)"""

    weight_format: WeightFormatArg = "all"
    """The weight format to limit testing to.

    (only relevant for model resources)"""

    devices: Optional[Union[str, Sequence[str]]] = None
    """Device(s) to use for testing"""

    decimal: int = 4
    """Precision for numerical comparisons"""

    def run(self):
        test(
            self.descr,
            weight_format=self.weight_format,
            devices=self.devices,
            decimal=self.decimal,
        )


class PackageCmd(CmdBase, WithSource):
    """bioimageio-package - save a resource's metadata with its associated files."""

    path: CliPositionalArg[Path]
    """The path to write the (zipped) package to.
    If it does not have a `.zip` suffix
    this command will save the package as an unzipped folder instead."""

    weight_format: WeightFormatArg = "all"
    """The weight format to include in the package (for model descriptions only)."""

    def run(self):
        if isinstance(self.descr, InvalidDescr):
            self.descr.validation_summary.display()
            raise ValueError("resource description is invalid")

        package(
            self.descr,
            self.path,
            weight_format=self.weight_format,
        )


def _get_stat(
    model_descr: AnyModelDescr,
    dataset: Iterable[Sample],
    dataset_length: int,
    stats_path: Path,
) -> Mapping[DatasetMeasure, MeasureValue]:
    req_dataset_meas, _ = get_required_dataset_measures(model_descr)
    if not req_dataset_meas:
        return {}

    req_dataset_meas, _ = get_required_dataset_measures(model_descr)
    stat_adapter = TypeAdapter(
        Mapping[DatasetMeasure, MeasureValue],
        config=ConfigDict(arbitrary_types_allowed=True),
    )

    if stats_path.exists():
        logger.info(f"loading precomputed dataset measures from {stats_path}")
        stat = stat_adapter.validate_json(stats_path.read_bytes())
        for m in req_dataset_meas:
            if m not in stat:
                raise ValueError(f"Missing {m} in {stats_path}")

        return stat

    stats_calc = StatsCalculator(req_dataset_meas)

    for sample in tqdm(
        dataset, total=dataset_length, descr="precomputing dataset stats", unit="sample"
    ):
        stats_calc.update(sample)

    stat = stats_calc.finalize()
    _ = stats_path.write_bytes(stat_adapter.dump_json(stat))

    return stat


class PredictCmd(CmdBase, WithSource):
    """bioimageio-predict - Run inference on your data with a bioimage.io model."""

    inputs: Union[str, Sequence[str]] = "model_inputs/*/{tensor_id}.*"
    """model inputs

    Either a single path/glob pattern including `{tensor_id}` to be used for all model inputs,
    or a list of paths/glob patterns for each model input respectively.

    For models with a single input a single path/glob pattern with `{tensor_id}` is also accepted.

    `.npy` and any file extension supported by imageio
    (listed at https://imageio.readthedocs.io/en/stable/formats/index.html#all-formats)
    are supported.
    """

    outputs: Union[str, Sequence[str]] = (
        "outputs_{model_id}/{sample_id}/{tensor_id}.npy"
    )
    """output paths analog to `inputs`"""

    overwrite: bool = False
    """allow overwriting existing output files"""

    blockwise: bool = False
    """process inputs blockwise"""

    stats: Path = Path("model_inputs/dataset_statistics.json")
    """path to dataset statistics
    (will be written if it does not exist,
    but the model requires statistical dataset measures)"""

    def run(self):
        model_descr = ensure_description_is_model(self.descr)

        input_ids = [
            t.name if isinstance(t, v0_4.InputTensorDescr) else t.id
            for t in model_descr.inputs
        ]
        output_ids = [
            t.name if isinstance(t, v0_4.OutputTensorDescr) else t.id
            for t in model_descr.outputs
        ]

        glob_matched_inputs: Dict[str, List[Path]] = {}
        n_glob_matches: Dict[int, List[str]] = {}

        if isinstance(self.inputs, str):
            if len(input_ids) > 1 and "{tensor_id}" not in self.inputs:
                raise ValueError(
                    f"{self.descr_id} needs inputs {input_ids}. Include '{{tensor_id}}' in `inputs` or provide multiple input paths/glob patterns."
                )

            inputs = [self.inputs.replace("{tensor_id}", t) for t in input_ids]
        else:
            inputs = self.inputs

        if len(inputs) < len(
            at_least := [
                str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
                for ipt in model_descr.inputs
                if not isinstance(ipt, v0_5.InputTensorDescr) or not ipt.optional
            ]
        ):
            raise ValueError(f"Expected at least {len(at_least)} inputs: {at_least}")

        if len(inputs) > len(
            at_most := [
                str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
                for ipt in model_descr.inputs
            ]
        ):
            raise ValueError(f"Expected at most {len(at_most)} inputs: {at_most}")

        input_patterns = [
            p.format(model_id=self.descr_id, tensor_id=t)
            for t, p in zip(input_ids, inputs)
        ]

        for input_id, pattern in zip(input_ids, input_patterns):
            paths = sorted(Path().glob(pattern))
            if not paths:
                raise FileNotFoundError(f"No file matched glob pattern '{pattern}'")

            glob_matched_inputs[input_id] = paths
            n_glob_matches.setdefault(len(paths), []).append(pattern)

        if len(n_glob_matches) > 1:
            raise ValueError(
                f"Different match counts for input glob patterns: '{n_glob_matches}'"
            )

        n_samples = list(n_glob_matches)[0]
        assert n_samples != 0, f"Did not find any input files at {n_glob_matches[0]}"

        # detect sample ids, assuming the default input pattern of `model-inputs/<sample_id>/<tensor_id>.ext`
        sample_ids: List[SampleId] = [
            p.parent.name for p in glob_matched_inputs[input_ids[0]]
        ]
        if len(sample_ids) != len(set(sample_ids)) or any(
            sample_ids[i] != p.parent.name
            for input_id in input_ids[1:]
            for i, p in enumerate(glob_matched_inputs[input_id])
        ):
            # fallback to sample1, sample2, ...
            digits = len(str(len(sample_ids) - 1))
            sample_ids = [f"sample{i:0{digits}}" for i in range(len(sample_ids))]

        if isinstance(self.outputs, str):
            if len(output_ids) > 1 and "{tensor_id}" not in self.outputs:
                raise ValueError(
                    f"{self.descr_id} produces outputs {output_ids}. Include '{{tensor_id}}' in `outputs` or provide {len(output_ids)} paths/patterns."
                )
            output_patterns = [
                self.outputs.replace("{tensor_id}", t) for t in output_ids
            ]
        elif len(self.outputs) != len(output_ids):
            raise ValueError(f"Expected {len(output_ids)} outputs: {output_ids}")
        else:
            output_patterns = self.outputs

        output_paths = {
            MemberId(t): [
                Path(
                    pattern.format(
                        model_id=self.descr_id,
                        i=i,
                        sample_id=sample_id,
                        tensor_id=t,
                    )
                )
                for i, sample_id in enumerate(sample_ids)
            ]
            for t, pattern in zip(output_ids, output_patterns)
        }
        if not self.overwrite:
            for paths in output_paths.values():
                for p in paths:
                    if p.exists():
                        raise FileExistsError(
                            f"{p} already exists. use --overwrite to (re-)write outputs anyway."
                        )

        def input_dataset(s: Stat):
            for i, sample_id in enumerate(sample_ids):
                yield load_sample_for_model(
                    model=model_descr,
                    paths={
                        MemberId(name): paths[i]
                        for name, paths in glob_matched_inputs.items()
                    },
                    stat=s,
                    sample_id=sample_id,
                )

        stat: Dict[Measure, MeasureValue] = {
            k: v
            for k, v in _get_stat(
                model_descr, input_dataset({}), len(sample_ids), self.stats
            ).items()
        }

        pp = create_prediction_pipeline(model_descr)
        predict_method = (
            pp.predict_sample_with_blocking
            if self.blockwise
            else pp.predict_sample_without_blocking
        )

        for i, input_sample in tqdm(
            enumerate(input_dataset(dict(stat))),
            total=n_samples,
            desc=f"predict with {self.descr_id}",
            unit="sample",
        ):
            output_sample = predict_method(input_sample)
            save_sample({m: output_paths[m][i] for m in output_paths}, output_sample)


class Bioimageio(
    BaseSettings,
    cli_parse_args=True,
    cli_prog_name="bioimageio",
    cli_use_class_docs_for_groups=True,
    use_attribute_docstrings=True,
):
    """bioimageio - CLI for bioimage.io resources 🦒"""

    model_config = SettingsConfigDict(
        json_file="bioimageio-cli.json", yaml_file="bioimageio-cli.yaml"
    )

    validate_format: CliSubCommand[ValidateFormatCmd]
    "Check a resource's metadata format"

    test: CliSubCommand[TestCmd]
    "Test a bioimageio resource (beyond meta data formatting)"

    package: CliSubCommand[PackageCmd]
    "Package a resource"

    predict: CliSubCommand[PredictCmd]
    "Predict with a model resource"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        cli: CliSettingsSource[BaseSettings] = CliSettingsSource(
            settings_cls, cli_parse_args=True
        )
        return (
            cli,
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            JsonConfigSettingsSource(settings_cls),
        )

    def run(self):
        cmd = self.validate_format or self.test or self.package or self.predict
        assert cmd is not None
        cmd.run()


assert isinstance(Bioimageio.__doc__, str)
Bioimageio.__doc__ += f"""

library versions:
  bioimageio.core {__version__}
  bioimageio.spec {__version__}

spec format versions:
        model RDF {ModelDescr.implemented_format_version}
      dataset RDF {DatasetDescr.implemented_format_version}
     notebook RDF {NotebookDescr.implemented_format_version}

"""
