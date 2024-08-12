import json
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from functools import cached_property
from pathlib import Path
from pprint import pformat, pprint
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from loguru import logger
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)
from pydantic.alias_generators import to_snake
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
from ruyaml import YAML
from tqdm import tqdm

from bioimageio.core import (
    MemberId,
    Sample,
    __version__,
    create_prediction_pipeline,
)
from bioimageio.core.commands import WeightFormatArg, package, test, validate_format
from bioimageio.core.common import SampleId
from bioimageio.core.digest_spec import get_member_ids, load_sample_for_model
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
from bioimageio.spec._internal.types import NotEmpty
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr, v0_4, v0_5
from bioimageio.spec.notebook import NotebookDescr
from bioimageio.spec.utils import download, ensure_description_is_model

yaml = YAML(typ="safe")


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

    inputs: NotEmpty[Sequence[Union[str, NotEmpty[Tuple[str, ...]]]]] = (
        "{input_id}/001.tif",
    )
    """Model input sample paths (for each input tensor).

    The input paths are expected to have shape...
     - `(n_samples,)` or `(n_samples,1)` for models expecting a single input tensor
     - `(n_samples,)` containing the substring '{input_id}', or
     - `(n_samples, n_model_inputs)` to provide each input tensor path explicitly.

    All substrings that are replaced by metadata from the model description:
    - '{model_id}'
    - '{input_id}'

    Example inputs to process sample 'a' and 'b'
    for a model expecting a 'raw' and a 'mask' input tensor:
    - `--inputs='[[a_raw.tif,a_mask.tif],[b_raw.tif,b_mask.tif]]'` (pure JSON style)
    - `--inputs a_raw.tif,a_mask.tif --inputs b_raw.tif,b_mask.tif` (Argparse + lazy style)
    - `--inputs='[a_raw.tif,a_mask.tif]','[b_raw.tif,b_mask.tif]'` (lazy + JSON style)
    (see https://docs.pydantic.dev/latest/concepts/pydantic_settings/#lists)
    Alternatively a `bioimageio-cli.yaml` (or `bioimageio-cli.json`) file may provide
    the arguments, e.g.:
    ```yaml
    inputs:
    - [a_raw.tif, a_mask.tif]
    - [b_raw.tif, b_mask.tif]
    ```

    `.npy` and any file extension supported by imageio are supported.
     Aavailable formats are listed at
    https://imageio.readthedocs.io/en/stable/formats/index.html#all-formats.
    Some formats have additional dependencies.
    """

    outputs: Union[str, NotEmpty[Tuple[str, ...]]] = (
        "outputs_{model_id}/{output_id}/{sample_id}.tif"
    )
    """Model output path pattern (per output tensor).

    All substrings that are replaced:
    - '{model_id}'
    - '{output_id}'
    - '{sample_id}'
    """

    overwrite: bool = False
    """allow overwriting existing output files"""

    blockwise: bool = False
    """process inputs blockwise"""

    stats: Path = Path("dataset_statistics.json")
    """path to dataset statistics
    (will be written if it does not exist,
    but the model requires statistical dataset measures)"""

    preview: bool = False
    """preview which files would be processed
    and what outputs would be generated."""

    example: bool = False
    """generate an example

    1. downloads example model inputs
    2. creates a `{model_id}_example` folder
    4. writes input arguments to `{model_id}_example/bioimageio-cli.yaml`
    5. executes a preview dry-run
    6. prints out the command line to run the prediction
    """

    def _example(self):
        model_descr = ensure_description_is_model(self.descr)
        input_ids = get_member_ids(model_descr.inputs)
        example_inputs = (
            model_descr.sample_inputs
            if isinstance(model_descr, v0_4.ModelDescr)
            else [ipt.sample_tensor or ipt.test_tensor for ipt in model_descr.inputs]
        )
        inputs001: List[str] = []
        example_path = Path(f"{self.descr_id}_example")

        for t, src in zip(input_ids, example_inputs):
            local = download(src).path
            dst = Path(f"{example_path}/{t}/001{''.join(local.suffixes)}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            inputs001.append(dst.as_posix())
            shutil.copy(local, dst)

        inputs = [tuple(inputs001)]
        output_pattern = f"{example_path}/outputs/{{output_id}}/{{sample_id}}.tif"
        bioimageio_cli_path = example_path / "bioimageio-cli.yaml"
        stats_file = "dataset_statistics.json"
        stats = (example_path / stats_file).as_posix()
        yaml.dump(
            dict(inputs=inputs, outputs=output_pattern, stats=stats_file),
            bioimageio_cli_path,
        )
        _ = subprocess.run(
            [
                "bioimageio",
                "predict",
                "--preview=True",  # update once we use implicit flags, see `class Bioimageio` below
                f"--stats='{stats}'",
                f"--inputs='{json.dumps(inputs)}'",
                f"--outputs='{output_pattern}'",
                f"'{self.source}'",
            ]
        )
        print(
            "run prediction of example input using the 'bioimageio-cli.yaml':\n"
            + f"cd {self.descr_id} && bioimageio predict '{self.source}'\n"
            + "Alternatively run the following command"
            + " (in the current workind directory, not the example folder):\n"
            + f"bioimageio predict --preview=False --stats='{stats}' --inputs='{json.dumps(inputs)}' --outputs='{output_pattern}' '{self.source}'"
        )

    def run(self):
        if self.example:
            return self._example()

        model_descr = ensure_description_is_model(self.descr)

        input_ids = get_member_ids(model_descr.inputs)
        output_ids = get_member_ids(model_descr.outputs)

        minimum_input_ids = tuple(
            str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
            for ipt in model_descr.inputs
            if not isinstance(ipt, v0_5.InputTensorDescr) or not ipt.optional
        )
        maximum_input_ids = tuple(
            str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
            for ipt in model_descr.inputs
        )

        def expand_inputs(i: int, ipt: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
            if isinstance(ipt, str):
                ipts = tuple(
                    ipt.format(model_id=self.descr_id, input_id=t) for t in input_ids
                )
            else:
                ipts = tuple(
                    p.format(model_id=self.descr_id, input_id=t)
                    for t, p in zip(input_ids, ipt)
                )

            if len(set(ipts)) < len(ipts):
                if len(minimum_input_ids) == len(maximum_input_ids):
                    n = len(minimum_input_ids)
                else:
                    n = f"{len(minimum_input_ids)}-{len(maximum_input_ids)}"

                raise ValueError(
                    f"[input sample #{i}] Include '{{input_id}}' in path pattern or explicitly specify {n} distinct input paths (got {ipt})"
                )

            if len(ipts) < len(minimum_input_ids):
                raise ValueError(
                    f"[input sample #{i}] Expected at least {len(minimum_input_ids)} inputs {minimum_input_ids}, got {ipts}"
                )

            if len(ipts) > len(maximum_input_ids):
                raise ValueError(
                    f"Expected at most {len(maximum_input_ids)} inputs {maximum_input_ids}, got {ipts}"
                )

            return ipts

        inputs = [expand_inputs(i, ipt) for i, ipt in enumerate(self.inputs, start=1)]

        sample_paths_in = [
            {t: Path(p) for t, p in zip(input_ids, ipts)} for ipts in inputs
        ]

        sample_ids = _get_sample_ids(sample_paths_in)

        def expand_outputs():
            if isinstance(self.outputs, str):
                outputs = [
                    tuple(
                        Path(
                            self.outputs.format(
                                model_id=self.descr_id, output_id=t, sample_id=s
                            )
                        )
                        for t in output_ids
                    )
                    for s in sample_ids
                ]
            else:
                outputs = [
                    tuple(
                        Path(p.format(model_id=self.descr_id, output_id=t, sample_id=s))
                        for t, p in zip(output_ids, self.outputs)
                    )
                    for s in sample_ids
                ]

            for i, out in enumerate(outputs, start=1):
                if len(set(out)) < len(out):
                    raise ValueError(
                        f"[output sample #{i}] Include '{{output_id}}' in path pattern or explicitly specify {len(output_ids)} distinct output paths (got {out})"
                    )

                if len(out) != len(output_ids):
                    raise ValueError(
                        f"[output sample #{i}] Expected {len(output_ids)} outputs {output_ids}, got {out}"
                    )

            return outputs

        outputs = expand_outputs()

        sample_paths_out = [
            {MemberId(t): Path(p) for t, p in zip(output_ids, out)} for out in outputs
        ]

        if not self.overwrite:
            for sample_paths in sample_paths_out:
                for p in sample_paths.values():
                    if p.exists():
                        raise FileExistsError(
                            f"{p} already exists. use --overwrite to (re-)write outputs anyway."
                        )
        if self.preview:
            pprint(
                {
                    "{sample_id}": dict(
                        inputs={"{input_id}": "<input path>"},
                        outputs={"{output_id}": "<output path>"},
                    )
                }
            )
            pprint(
                {
                    s: dict(
                        inputs={t: p.as_posix() for t, p in sp_in.items()},
                        outputs={t: p.as_posix() for t, p in sp_out.items()},
                    )
                    for s, sp_in, sp_out in zip(
                        sample_ids, sample_paths_in, sample_paths_out
                    )
                }
            )
            return

        def input_dataset(stat: Stat):
            for s, sp_in in zip(sample_ids, sample_paths_in):
                yield load_sample_for_model(
                    model=model_descr,
                    paths=sp_in,
                    stat=stat,
                    sample_id=s,
                )

        stat: Dict[Measure, MeasureValue] = dict(
            _get_stat(
                model_descr, input_dataset({}), len(sample_ids), self.stats
            ).items()
        )

        pp = create_prediction_pipeline(model_descr)
        predict_method = (
            pp.predict_sample_with_blocking
            if self.blockwise
            else pp.predict_sample_without_blocking
        )

        for sample_in, sp_out in tqdm(
            zip(input_dataset(dict(stat)), sample_paths_out),
            total=len(inputs),
            desc=f"predict with {self.descr_id}",
            unit="sample",
        ):
            sample_out = predict_method(sample_in)
            save_sample(sp_out, sample_out)


class Bioimageio(
    BaseSettings,
    # alias_generator=AliasGenerator(
    #     validation_alias=lambda s: AliasChoices(s, to_snake(s).replace("_", "-"))
    # ),
    # TODO: investigate how to allow a validation alias for subcommands
    #       ('validate-format' vs 'validate_format')
    cli_parse_args=True,
    cli_prog_name="bioimageio",
    cli_use_class_docs_for_groups=True,
    # cli_implicit_flags=True, # TODO: make flags implicit, see https://github.com/pydantic/pydantic-settings/issues/361
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
        sys_args = pformat(sys.argv)
        logger.info("starting CLI with arguments:\n{}", sys_args)
        return (
            cli,
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            JsonConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="before")
    @classmethod
    def _log(cls, data: Any):
        logger.info(
            "loaded CLI input:\n{}",
            pformat({k: v for k, v in data.items() if v is not None}),
        )
        return data

    def run(self):
        logger.info(
            "executing CLI command:\n{}",
            pformat({k: v for k, v in self.model_dump().items() if v is not None}),
        )
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


def _get_sample_ids(
    input_paths: Sequence[Mapping[MemberId, Path]]
) -> Sequence[SampleId]:
    """Get sample ids for given input paths, based on the common path per sample.

    Falls back to sample01, samle02, etc..."""

    matcher = SequenceMatcher()

    def get_common_seq(seqs: Sequence[Sequence[str]]) -> Sequence[str]:
        """extract a common sequence from multiple sequences
        (order sensitive; strips whitespace and slashes)
        """
        common = seqs[0]

        for seq in seqs[1:]:
            if not seq:
                continue
            matcher.set_seqs(common, seq)
            i, _, size = matcher.find_longest_match()
            common = common[i : i + size]

        if isinstance(common, str):
            common = common.strip().strip("/")
        else:
            common = [cs for c in common if (cs := c.strip().strip("/"))]

        if not common:
            raise ValueError(f"failed to find common sequence for {seqs}")

        return common

    def get_shorter_diff(seqs: Sequence[Sequence[str]]) -> List[Sequence[str]]:
        """get a shorter sequence whose entries are still unique
        (order sensitive, not minimal sequence)
        """
        min_seq_len = min(len(s) for s in seqs)
        # cut from the start
        for start in range(min_seq_len - 1, -1, -1):
            shortened = [s[start:] for s in seqs]
            if len(set(shortened)) == len(seqs):
                min_seq_len -= start
                break
        else:
            seen: Set[Sequence[str]] = set()
            dupes = [s for s in seqs if s in seen or seen.add(s)]
            raise ValueError(f"Found duplicate entries {dupes}")

        # cut from the end
        for end in range(min_seq_len - 1, 1, -1):
            shortened = [s[:end] for s in shortened]
            if len(set(shortened)) == len(seqs):
                break

        return shortened

    full_tensor_ids = [
        sorted(
            p.resolve().with_suffix("").as_posix() for p in input_sample_paths.values()
        )
        for input_sample_paths in input_paths
    ]
    try:
        long_sample_ids = [get_common_seq(t) for t in full_tensor_ids]
        sample_ids = get_shorter_diff(long_sample_ids)
    except ValueError as e:
        raise ValueError(f"failed to extract sample ids: {e}")

    return sample_ids
