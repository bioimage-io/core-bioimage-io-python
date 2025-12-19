import hashlib
import os
import platform
import subprocess
import sys
import warnings
from io import StringIO
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict, Unpack, assert_never, get_args

from bioimageio.spec import (
    AnyDatasetDescr,
    AnyModelDescr,
    BioimageioCondaEnv,
    DatasetDescr,
    InvalidDescr,
    LatestResourceDescr,
    ModelDescr,
    ResourceDescr,
    ValidationContext,
    build_description,
    dump_description,
    get_conda_env,
    load_description,
    save_bioimageio_package,
)
from bioimageio.spec._description_impl import DISCOVER
from bioimageio.spec._internal.common_nodes import ResourceDescrBase
from bioimageio.spec._internal.io import is_yaml_value
from bioimageio.spec._internal.io_utils import read_yaml, write_yaml
from bioimageio.spec._internal.types import (
    AbsoluteTolerance,
    FormatVersionPlaceholder,
    MismatchedElementsPerMillion,
    RelativeTolerance,
)
from bioimageio.spec._internal.validation_context import get_validation_context
from bioimageio.spec.common import BioimageioYamlContent, PermissiveFileSource, Sha256
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import WeightsFormat
from bioimageio.spec.summary import (
    ErrorEntry,
    InstalledPackage,
    ValidationDetail,
    ValidationSummary,
    WarningEntry,
)

from . import __version__
from ._prediction_pipeline import create_prediction_pipeline
from ._settings import settings
from .axis import AxisId, BatchSize
from .common import MemberId, SupportedWeightsFormat
from .digest_spec import get_test_input_sample, get_test_output_sample
from .io import save_tensor
from .sample import Sample

CONDA_CMD = "conda.bat" if platform.system() == "Windows" else "conda"


class DeprecatedKwargs(TypedDict):
    absolute_tolerance: NotRequired[AbsoluteTolerance]
    relative_tolerance: NotRequired[RelativeTolerance]
    decimal: NotRequired[Optional[int]]


def enable_determinism(
    mode: Literal["seed_only", "full"] = "full",
    weight_formats: Optional[Sequence[SupportedWeightsFormat]] = None,
):
    """Seed and configure ML frameworks for maximum reproducibility.
    May degrade performance. Only recommended for testing reproducibility!

    Seed any random generators and (if **mode**=="full") request ML frameworks to use
    deterministic algorithms.

    Args:
        mode: determinism mode
            - 'seed_only' -- only set seeds, or
            - 'full' determinsm features (might degrade performance or throw exceptions)
        weight_formats: Limit deep learning importing deep learning frameworks
            based on weight_formats.
            E.g. this allows to avoid importing tensorflow when testing with pytorch.

    Notes:
        - **mode** == "full"  might degrade performance or throw exceptions.
        - Subsequent inference calls might still differ. Call before each function
          (sequence) that is expected to be reproducible.
        - Degraded performance: Use for testing reproducibility only!
        - Recipes:
            - [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
            - [Keras](https://keras.io/examples/keras_recipes/reproducibility_recipes/)
            - [NumPy](https://numpy.org/doc/2.0/reference/random/generated/numpy.random.seed.html)
    """
    try:
        try:
            import numpy.random
        except ImportError:
            pass
        else:
            numpy.random.seed(0)
    except Exception as e:
        logger.debug(str(e))

    if (
        weight_formats is None
        or "pytorch_state_dict" in weight_formats
        or "torchscript" in weight_formats
    ):
        try:
            try:
                import torch
            except ImportError:
                pass
            else:
                _ = torch.manual_seed(0)
                torch.use_deterministic_algorithms(mode == "full")
        except Exception as e:
            logger.debug(str(e))

    if (
        weight_formats is None
        or "tensorflow_saved_model_bundle" in weight_formats
        or "keras_hdf5" in weight_formats
    ):
        try:
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            try:
                import tensorflow as tf  # pyright: ignore[reportMissingTypeStubs]
            except ImportError:
                pass
            else:
                tf.random.set_seed(0)
                if mode == "full":
                    tf.config.experimental.enable_op_determinism()
                # TODO: find possibility to switch it off again??
        except Exception as e:
            logger.debug(str(e))

    if weight_formats is None or "keras_hdf5" in weight_formats:
        try:
            try:
                import keras  # pyright: ignore[reportMissingTypeStubs]
            except ImportError:
                pass
            else:
                keras.utils.set_random_seed(0)
        except Exception as e:
            logger.debug(str(e))


def test_model(
    source: Union[v0_4.ModelDescr, v0_5.ModelDescr, PermissiveFileSource],
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[List[str]] = None,
    *,
    determinism: Literal["seed_only", "full"] = "seed_only",
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> ValidationSummary:
    """Test model inference"""
    return test_description(
        source,
        weight_format=weight_format,
        devices=devices,
        determinism=determinism,
        expected_type="model",
        sha256=sha256,
        stop_early=stop_early,
        **deprecated,
    )


def default_run_command(args: Sequence[str]):
    logger.info("running '{}'...", " ".join(args))
    _ = subprocess.check_call(args)


def test_description(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[FormatVersionPlaceholder, str] = "discover",
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Optional[str] = None,
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    runtime_env: Union[
        Literal["currently-active", "as-described"], Path, BioimageioCondaEnv
    ] = ("currently-active"),
    run_command: Callable[[Sequence[str]], None] = default_run_command,
    **deprecated: Unpack[DeprecatedKwargs],
) -> ValidationSummary:
    """Test a bioimage.io resource dynamically,
    for example run prediction of test tensors for models.

    Args:
        source: model description source.
        weight_format: Weight format to test.
            Default: All weight formats present in **source**.
        devices: Devices to test with, e.g. 'cpu', 'cuda'.
            Default (may be weight format dependent): ['cuda'] if available, ['cpu'] otherwise.
        determinism: Modes to improve reproducibility of test outputs.
        expected_type: Assert an expected resource description `type`.
        sha256: Expected SHA256 value of **source**.
                (Ignored if **source** already is a loaded `ResourceDescr` object.)
        stop_early: Do not run further subtests after a failed one.
        runtime_env: (Experimental feature!) The Python environment to run the tests in
            - `"currently-active"`: Use active Python interpreter.
            - `"as-described"`: Use `bioimageio.spec.get_conda_env` to generate a conda
                environment YAML file based on the model weights description.
            - A `BioimageioCondaEnv` or a path to a conda environment YAML file.
                Note: The `bioimageio.core` dependency will be added automatically if not present.
        run_command: (Experimental feature!) Function to execute (conda) terminal commands in a subprocess.
            The function should raise an exception if the command fails.
            **run_command** is ignored if **runtime_env** is `"currently-active"`.
    """
    if runtime_env == "currently-active":
        rd = load_description_and_test(
            source,
            format_version=format_version,
            weight_format=weight_format,
            devices=devices,
            determinism=determinism,
            expected_type=expected_type,
            sha256=sha256,
            stop_early=stop_early,
            **deprecated,
        )
        return rd.validation_summary

    if runtime_env == "as-described":
        conda_env = None
    elif isinstance(runtime_env, (str, Path)):
        conda_env = BioimageioCondaEnv.model_validate(read_yaml(Path(runtime_env)))
    elif isinstance(runtime_env, BioimageioCondaEnv):
        conda_env = runtime_env
    else:
        assert_never(runtime_env)

    try:
        run_command(["thiscommandshouldalwaysfail", "please"])
    except Exception:
        pass
    else:
        raise RuntimeError(
            "given run_command does not raise an exception for a failing command"
        )

    td_kwargs: Dict[str, Any] = (
        dict(ignore_cleanup_errors=True) if sys.version_info >= (3, 10) else {}
    )
    with TemporaryDirectory(**td_kwargs) as _d:
        working_dir = Path(_d)

        if isinstance(source, ResourceDescrBase):
            descr = source
        elif isinstance(source, dict):
            context = get_validation_context().replace(
                perform_io_checks=True  # make sure we perform io checks though
            )

            descr = build_description(source, context=context)
        else:
            descr = load_description(source, perform_io_checks=True)

        if isinstance(descr, InvalidDescr):
            return descr.validation_summary
        elif isinstance(source, (dict, ResourceDescrBase)):
            file_source = save_bioimageio_package(
                descr, output_path=working_dir / "package.zip"
            )
        else:
            file_source = source

        _test_in_env(
            file_source,
            descr=descr,
            working_dir=working_dir,
            weight_format=weight_format,
            conda_env=conda_env,
            devices=devices,
            determinism=determinism,
            expected_type=expected_type,
            sha256=sha256,
            stop_early=stop_early,
            run_command=run_command,
            **deprecated,
        )

    return descr.validation_summary


def _test_in_env(
    source: PermissiveFileSource,
    *,
    descr: ResourceDescr,
    working_dir: Path,
    weight_format: Optional[SupportedWeightsFormat],
    conda_env: Optional[BioimageioCondaEnv],
    devices: Optional[Sequence[str]],
    determinism: Literal["seed_only", "full"],
    run_command: Callable[[Sequence[str]], None],
    stop_early: bool,
    expected_type: Optional[str],
    sha256: Optional[Sha256],
    **deprecated: Unpack[DeprecatedKwargs],
):
    """Test a bioimage.io resource in a given conda environment.
    Adds details to the existing validation summary of **descr**.
    """
    if isinstance(descr, (v0_4.ModelDescr, v0_5.ModelDescr)):
        if weight_format is None:
            # run tests for all present weight formats
            all_present_wfs = [
                wf for wf in get_args(WeightsFormat) if getattr(descr.weights, wf)
            ]
            ignore_wfs = [wf for wf in all_present_wfs if wf in ["tensorflow_js"]]
            logger.info(
                "Found weight formats {}. Start testing all{}...",
                all_present_wfs,
                f" (except: {', '.join(ignore_wfs)}) " if ignore_wfs else "",
            )
            for wf in all_present_wfs:
                _test_in_env(
                    source,
                    descr=descr,
                    working_dir=working_dir / wf,
                    weight_format=wf,
                    devices=devices,
                    determinism=determinism,
                    conda_env=conda_env,
                    run_command=run_command,
                    expected_type=expected_type,
                    sha256=sha256,
                    stop_early=stop_early,
                    **deprecated,
                )

            return

        if weight_format == "pytorch_state_dict":
            wf = descr.weights.pytorch_state_dict
        elif weight_format == "torchscript":
            wf = descr.weights.torchscript
        elif weight_format == "keras_hdf5":
            wf = descr.weights.keras_hdf5
        elif weight_format == "onnx":
            wf = descr.weights.onnx
        elif weight_format == "tensorflow_saved_model_bundle":
            wf = descr.weights.tensorflow_saved_model_bundle
        elif weight_format == "tensorflow_js":
            raise RuntimeError(
                "testing 'tensorflow_js' is not supported by bioimageio.core"
            )
        else:
            assert_never(weight_format)
        assert wf is not None
        if conda_env is None:
            conda_env = get_conda_env(entry=wf)

        test_loc = ("weights", weight_format)
    else:
        if conda_env is None:
            warnings.warn(
                "No conda environment description given for testing (And no default conda envs available for non-model descriptions)."
            )
            return

        test_loc = ()

    # remove name as we crate a name based on the env description hash value
    conda_env.name = None

    dumped_env = conda_env.model_dump(mode="json", exclude_none=True)
    if not is_yaml_value(dumped_env):
        raise ValueError(f"Failed to dump conda env to valid YAML {conda_env}")

    env_io = StringIO()
    write_yaml(dumped_env, file=env_io)
    encoded_env = env_io.getvalue().encode()
    env_name = hashlib.sha256(encoded_env).hexdigest()

    try:
        run_command(["where" if platform.system() == "Windows" else "which", CONDA_CMD])
    except Exception as e:
        raise RuntimeError("Conda not available") from e

    try:
        run_command([CONDA_CMD, "run", "-n", env_name, "python", "--version"])
    except Exception as e:
        working_dir.mkdir(parents=True, exist_ok=True)
        path = working_dir / "env.yaml"
        try:
            _ = path.write_bytes(encoded_env)
            logger.debug("written conda env to {}", path)
            run_command(
                [
                    CONDA_CMD,
                    "env",
                    "create",
                    "--yes",
                    f"--file={path}",
                    f"--name={env_name}",
                ]
                + (["--quiet"] if settings.CI else [])
            )
            # double check that environment was created successfully
            run_command([CONDA_CMD, "run", "-n", env_name, "python", "--version"])
        except Exception as e:
            descr.validation_summary.add_detail(
                ValidationDetail(
                    name="Conda environment creation",
                    status="failed",
                    loc=test_loc,
                    recommended_env=conda_env,
                    errors=[
                        ErrorEntry(
                            loc=test_loc,
                            msg=str(e),
                            type="conda",
                            with_traceback=True,
                        )
                    ],
                )
            )
            return

    working_dir.mkdir(parents=True, exist_ok=True)
    summary_path = working_dir / "summary.json"
    assert not summary_path.exists(), "Summary file already exists"
    cmd = []
    cmd_error = None
    for summary_path_arg_name in ("summary", "summary-path"):
        try:
            run_command(
                cmd := (
                    [
                        CONDA_CMD,
                        "run",
                        "-n",
                        env_name,
                        "bioimageio",
                        "test",
                        str(source),
                        f"--{summary_path_arg_name}={summary_path.as_posix()}",
                        f"--determinism={determinism}",
                    ]
                    + ([f"--expected-type={expected_type}"] if expected_type else [])
                    + (["--stop-early"] if stop_early else [])
                )
            )
        except Exception as e:
            cmd_error = f"Command '{' '.join(cmd)}' returned with error: {e}."

        if summary_path.exists():
            break
    else:
        if cmd_error is not None:
            logger.warning(cmd_error)

        descr.validation_summary.add_detail(
            ValidationDetail(
                name="run 'bioimageio test' command",
                recommended_env=conda_env,
                errors=[
                    ErrorEntry(
                        loc=(),
                        type="bioimageio cli",
                        msg=f"test command '{' '.join(cmd)}' did not produce a summary file at {summary_path}",
                    )
                ],
                status="failed",
            )
        )
        return

    # add relevant details from command summary
    command_summary = ValidationSummary.load_json(summary_path)
    for detail in command_summary.details:
        if detail.loc[: len(test_loc)] == test_loc:
            descr.validation_summary.add_detail(detail)


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Literal["latest"],
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Literal["model"],
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[ModelDescr, InvalidDescr]: ...


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Literal["latest"],
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Literal["dataset"],
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[DatasetDescr, InvalidDescr]: ...


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Literal["latest"],
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Optional[str] = None,
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[LatestResourceDescr, InvalidDescr]: ...


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[FormatVersionPlaceholder, str] = DISCOVER,
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Literal["model"],
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[AnyModelDescr, InvalidDescr]: ...


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[FormatVersionPlaceholder, str] = DISCOVER,
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Literal["dataset"],
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[AnyDatasetDescr, InvalidDescr]: ...


@overload
def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[FormatVersionPlaceholder, str] = DISCOVER,
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Optional[str] = None,
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[ResourceDescr, InvalidDescr]: ...


def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[FormatVersionPlaceholder, str] = DISCOVER,
    weight_format: Optional[SupportedWeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    determinism: Literal["seed_only", "full"] = "seed_only",
    expected_type: Optional[str] = None,
    sha256: Optional[Sha256] = None,
    stop_early: bool = True,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Union[ResourceDescr, InvalidDescr]:
    """Test a bioimage.io resource dynamically,
    for example run prediction of test tensors for models.

    See `test_description` for more details.

    Returns:
        A (possibly invalid) resource description object
        with a populated `.validation_summary` attribute.
    """
    if isinstance(source, ResourceDescrBase):
        root = source.root
        file_name = source.file_name
        if (
            (
                format_version
                not in (
                    DISCOVER,
                    source.format_version,
                    ".".join(source.format_version.split(".")[:2]),
                )
            )
            or (c := source.validation_summary.details[0].context) is None
            or not c.perform_io_checks
        ):
            logger.debug(
                "deserializing source to ensure we validate and test using format {} and perform io checks",
                format_version,
            )
            source = dump_description(source)
    else:
        root = Path()
        file_name = None

    if isinstance(source, ResourceDescrBase):
        rd = source
    elif isinstance(source, dict):
        # check context for a given root; default to root of source
        context = get_validation_context(
            ValidationContext(root=root, file_name=file_name)
        ).replace(
            perform_io_checks=True  # make sure we perform io checks though
        )

        rd = build_description(
            source,
            format_version=format_version,
            context=context,
        )
    else:
        rd = load_description(
            source, format_version=format_version, sha256=sha256, perform_io_checks=True
        )

    rd.validation_summary.env.add(
        InstalledPackage(name="bioimageio.core", version=__version__)
    )

    if expected_type is not None:
        _test_expected_resource_type(rd, expected_type)

    if isinstance(rd, (v0_4.ModelDescr, v0_5.ModelDescr)):
        if weight_format is None:
            weight_formats: List[SupportedWeightsFormat] = [
                w for w, we in rd.weights if we is not None
            ]  # pyright: ignore[reportAssignmentType]
        else:
            weight_formats = [weight_format]

        enable_determinism(determinism, weight_formats=weight_formats)
        for w in weight_formats:
            _test_model_inference(rd, w, devices, stop_early=stop_early, **deprecated)
            if stop_early and rd.validation_summary.status != "passed":
                break

            if not isinstance(rd, v0_4.ModelDescr):
                _test_model_inference_parametrized(
                    rd, w, devices, stop_early=stop_early
                )
                if stop_early and rd.validation_summary.status != "passed":
                    break

    # TODO: add execution of jupyter notebooks
    # TODO: add more tests

    return rd


def _get_tolerance(
    model: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    wf: SupportedWeightsFormat,
    m: MemberId,
    **deprecated: Unpack[DeprecatedKwargs],
) -> Tuple[RelativeTolerance, AbsoluteTolerance, MismatchedElementsPerMillion]:
    if isinstance(model, v0_5.ModelDescr):
        applicable = v0_5.ReproducibilityTolerance()

        # check legacy test kwargs for weight format specific tolerance
        if model.config.bioimageio.model_extra is not None:
            for weights_format, test_kwargs in model.config.bioimageio.model_extra.get(
                "test_kwargs", {}
            ).items():
                if wf == weights_format:
                    applicable = v0_5.ReproducibilityTolerance(
                        relative_tolerance=test_kwargs.get("relative_tolerance", 1e-3),
                        absolute_tolerance=test_kwargs.get("absolute_tolerance", 1e-3),
                    )
                    break

        # check for weights format and output tensor specific tolerance
        for a in model.config.bioimageio.reproducibility_tolerance:
            if (not a.weights_formats or wf in a.weights_formats) and (
                not a.output_ids or m in a.output_ids
            ):
                applicable = a
                break

        rtol = applicable.relative_tolerance
        atol = applicable.absolute_tolerance
        mismatched_tol = applicable.mismatched_elements_per_million
    elif (decimal := deprecated.get("decimal")) is not None:
        warnings.warn(
            "The argument `decimal` has been deprecated in favour of"
            + " `relative_tolerance` and `absolute_tolerance`, with different"
            + " validation logic, using `numpy.testing.assert_allclose, see"
            + " 'https://numpy.org/doc/stable/reference/generated/"
            + " numpy.testing.assert_allclose.html'. Passing a value for `decimal`"
            + " will cause validation to revert to the old behaviour."
        )
        atol = 1.5 * 10 ** (-decimal)
        rtol = 0
        mismatched_tol = 0
    else:
        # use given (deprecated) test kwargs
        atol = deprecated.get("absolute_tolerance", 1e-3)
        rtol = deprecated.get("relative_tolerance", 1e-3)
        mismatched_tol = 0

    return rtol, atol, mismatched_tol


def _test_model_inference(
    model: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    weight_format: SupportedWeightsFormat,
    devices: Optional[Sequence[str]],
    stop_early: bool,
    **deprecated: Unpack[DeprecatedKwargs],
) -> None:
    test_name = f"Reproduce test outputs from test inputs ({weight_format})"
    logger.debug("starting '{}'", test_name)
    error_entries: List[ErrorEntry] = []
    warning_entries: List[WarningEntry] = []

    def add_error_entry(msg: str, with_traceback: bool = False):
        error_entries.append(
            ErrorEntry(
                loc=("weights", weight_format),
                msg=msg,
                type="bioimageio.core",
                with_traceback=with_traceback,
            )
        )

    def add_warning_entry(msg: str):
        warning_entries.append(
            WarningEntry(
                loc=("weights", weight_format),
                msg=msg,
                type="bioimageio.core",
            )
        )

    try:
        test_input = get_test_input_sample(model)
        expected = get_test_output_sample(model)

        with create_prediction_pipeline(
            bioimageio_model=model, devices=devices, weight_format=weight_format
        ) as prediction_pipeline:
            results = prediction_pipeline.predict_sample_without_blocking(test_input)

        if len(results.members) != len(expected.members):
            add_error_entry(
                f"Expected {len(expected.members)} outputs, but got {len(results.members)}"
            )

        else:
            for m, expected in expected.members.items():
                actual = results.members.get(m)
                if actual is None:
                    add_error_entry("Output tensors for test case may not be None")
                    if stop_early:
                        break
                    else:
                        continue

                if actual.dims != (dims := expected.dims):
                    add_error_entry(
                        f"Output '{m}' has dims {actual.dims}, but expected {expected.dims}"
                    )
                    if stop_early:
                        break
                    else:
                        continue

                if actual.tagged_shape != expected.tagged_shape:
                    add_error_entry(
                        f"Output '{m}' has shape {actual.tagged_shape}, but expected {expected.tagged_shape}"
                    )
                    if stop_early:
                        break
                    else:
                        continue

                try:
                    expected_np = expected.data.to_numpy().astype(np.float32)
                    del expected
                    actual_np: NDArray[Any] = actual.data.to_numpy().astype(np.float32)

                    rtol, atol, mismatched_tol = _get_tolerance(
                        model, wf=weight_format, m=m, **deprecated
                    )
                    rtol_value = rtol * abs(expected_np)
                    abs_diff = abs(actual_np - expected_np)
                    mismatched = abs_diff > atol + rtol_value
                    mismatched_elements = mismatched.sum().item()
                    if not mismatched_elements:
                        continue

                    actual_output_path = Path(f"actual_output_{m}_{weight_format}.npy")
                    try:
                        save_tensor(actual_output_path, actual)
                    except Exception as e:
                        logger.error(
                            "Failed to save actual output tensor to {}: {}",
                            actual_output_path,
                            e,
                        )

                    mismatched_ppm = mismatched_elements / expected_np.size * 1e6
                    abs_diff[~mismatched] = 0  # ignore non-mismatched elements

                    r_max_idx_flat = (
                        r_diff := (abs_diff / (abs(expected_np) + 1e-6))
                    ).argmax()
                    r_max_idx = np.unravel_index(r_max_idx_flat, r_diff.shape)
                    r_max = r_diff[r_max_idx].item()
                    r_actual = actual_np[r_max_idx].item()
                    r_expected = expected_np[r_max_idx].item()

                    # Calculate the max absolute difference with the relative tolerance subtracted
                    abs_diff_wo_rtol: NDArray[np.float32] = abs_diff - rtol_value
                    a_max_idx = np.unravel_index(
                        abs_diff_wo_rtol.argmax(), abs_diff_wo_rtol.shape
                    )

                    a_max = abs_diff[a_max_idx].item()
                    a_actual = actual_np[a_max_idx].item()
                    a_expected = expected_np[a_max_idx].item()
                except Exception as e:
                    msg = f"Output '{m}' disagrees with expected values."
                    add_error_entry(msg)
                    if stop_early:
                        break
                else:
                    msg = (
                        f"Output '{m}' disagrees with {mismatched_elements} of"
                        + f" {expected_np.size} expected values"
                        + f" ({mismatched_ppm:.1f} ppm)."
                        + f"\n Max relative difference not accounted for by absolute tolerance ({atol:.2e}): {r_max:.2e}"
                        + rf" (= \|{r_actual:.2e} - {r_expected:.2e}\|/\|{r_expected:.2e} + 1e-6\|)"
                        + f" at {dict(zip(dims, r_max_idx))}"
                        + f"\n Max absolute difference not accounted for by relative tolerance ({rtol:.2e}): {a_max:.2e}"
                        + rf" (= \|{a_actual:.7e} - {a_expected:.7e}\|) at {dict(zip(dims, a_max_idx))}"
                        + f"\n Saved actual output to {actual_output_path}."
                    )
                    if mismatched_ppm > mismatched_tol:
                        add_error_entry(msg)
                        if stop_early:
                            break
                    else:
                        add_warning_entry(msg)

    except Exception as e:
        if get_validation_context().raise_errors:
            raise e

        add_error_entry(str(e), with_traceback=True)

    model.validation_summary.add_detail(
        ValidationDetail(
            name=test_name,
            loc=("weights", weight_format),
            status="failed" if error_entries else "passed",
            recommended_env=get_conda_env(entry=dict(model.weights)[weight_format]),
            errors=error_entries,
            warnings=warning_entries,
        )
    )


def _test_model_inference_parametrized(
    model: v0_5.ModelDescr,
    weight_format: SupportedWeightsFormat,
    devices: Optional[Sequence[str]],
    *,
    stop_early: bool,
) -> None:
    if not any(
        isinstance(a.size, v0_5.ParameterizedSize)
        for ipt in model.inputs
        for a in ipt.axes
    ):
        # no parameterized sizes => set n=0
        ns: Set[v0_5.ParameterizedSize_N] = {0}
    else:
        ns = {0, 1, 2}

    given_batch_sizes = {
        a.size
        for ipt in model.inputs
        for a in ipt.axes
        if isinstance(a, v0_5.BatchAxis)
    }
    if given_batch_sizes:
        batch_sizes = {gbs for gbs in given_batch_sizes if gbs is not None}
        if not batch_sizes:
            # only arbitrary batch sizes
            batch_sizes = {1, 2}
    else:
        # no batch axis
        batch_sizes = {1}

    test_cases: Set[Tuple[BatchSize, v0_5.ParameterizedSize_N]] = {
        (b, n) for b, n in product(sorted(batch_sizes), sorted(ns))
    }
    logger.info(
        "Testing inference with '{}' for {} different inputs (B, N): {}",
        weight_format,
        len(test_cases),
        test_cases,
    )

    def generate_test_cases():
        tested: Set[Hashable] = set()

        def get_ns(n: int):
            return {
                (t.id, a.id): n
                for t in model.inputs
                for a in t.axes
                if isinstance(a.size, v0_5.ParameterizedSize)
            }

        for batch_size, n in sorted(test_cases):
            input_target_sizes, expected_output_sizes = model.get_axis_sizes(
                get_ns(n), batch_size=batch_size
            )
            hashable_target_size = tuple(
                (k, input_target_sizes[k]) for k in sorted(input_target_sizes)
            )
            if hashable_target_size in tested:
                continue
            else:
                tested.add(hashable_target_size)

            resized_test_inputs = Sample(
                members={
                    t.id: (
                        test_input.members[t.id].resize_to(
                            {
                                aid: s
                                for (tid, aid), s in input_target_sizes.items()
                                if tid == t.id
                            },
                        )
                    )
                    for t in model.inputs
                },
                stat=test_input.stat,
                id=test_input.id,
            )
            expected_output_shapes = {
                t.id: {
                    aid: s
                    for (tid, aid), s in expected_output_sizes.items()
                    if tid == t.id
                }
                for t in model.outputs
            }
            yield n, batch_size, resized_test_inputs, expected_output_shapes

    try:
        test_input = get_test_input_sample(model)

        with create_prediction_pipeline(
            bioimageio_model=model, devices=devices, weight_format=weight_format
        ) as prediction_pipeline:
            for n, batch_size, inputs, exptected_output_shape in generate_test_cases():
                error: Optional[str] = None
                try:
                    result = prediction_pipeline.predict_sample_without_blocking(inputs)
                except Exception as e:
                    error = str(e)
                else:
                    if len(result.members) != len(exptected_output_shape):
                        error = (
                            f"Expected {len(exptected_output_shape)} outputs,"
                            + f" but got {len(result.members)}"
                        )

                    else:
                        for m, exp in exptected_output_shape.items():
                            res = result.members.get(m)
                            if res is None:
                                error = "Output tensors may not be None for test case"
                                break

                            diff: Dict[AxisId, int] = {}
                            for a, s in res.sizes.items():
                                if isinstance((e_aid := exp[AxisId(a)]), int):
                                    if s != e_aid:
                                        diff[AxisId(a)] = s
                                elif (
                                    s < e_aid.min
                                    or e_aid.max is not None
                                    and s > e_aid.max
                                ):
                                    diff[AxisId(a)] = s
                            if diff:
                                error = (
                                    f"(n={n}) Expected output shape {exp},"
                                    + f" but got {res.sizes} (diff: {diff})"
                                )
                                break

                model.validation_summary.add_detail(
                    ValidationDetail(
                        name=f"Run {weight_format} inference for inputs with"
                        + f" batch_size: {batch_size} and size parameter n: {n}",
                        loc=("weights", weight_format),
                        status="passed" if error is None else "failed",
                        errors=(
                            []
                            if error is None
                            else [
                                ErrorEntry(
                                    loc=("weights", weight_format),
                                    msg=error,
                                    type="bioimageio.core",
                                )
                            ]
                        ),
                    )
                )
                if stop_early and error is not None:
                    break
    except Exception as e:
        if get_validation_context().raise_errors:
            raise e

        model.validation_summary.add_detail(
            ValidationDetail(
                name=f"Run {weight_format} inference for parametrized inputs",
                status="failed",
                loc=("weights", weight_format),
                errors=[
                    ErrorEntry(
                        loc=("weights", weight_format),
                        msg=str(e),
                        type="bioimageio.core",
                        with_traceback=True,
                    )
                ],
            )
        )


def _test_expected_resource_type(
    rd: Union[InvalidDescr, ResourceDescr], expected_type: str
):
    has_expected_type = rd.type == expected_type
    rd.validation_summary.details.append(
        ValidationDetail(
            name="Has expected resource type",
            status="passed" if has_expected_type else "failed",
            loc=("type",),
            errors=(
                []
                if has_expected_type
                else [
                    ErrorEntry(
                        loc=("type",),
                        type="type",
                        msg=f"Expected type {expected_type}, found {rd.type}",
                    )
                ]
            ),
        )
    )


# TODO: Implement `debug_model()`
# def debug_model(
#     model_rdf: Union[RawResourceDescr, ResourceDescr, URI, Path, str],
#     *,
#     weight_format: Optional[WeightsFormat] = None,
#     devices: Optional[List[str]] = None,
# ):
#     """Run the model test and return dict with inputs, results, expected results and intermediates.

#     Returns dict with tensors "inputs", "inputs_processed", "outputs_raw", "outputs", "expected" and "diff".
#     """
#     inputs_raw: Optional = None
#     inputs_processed: Optional = None
#     outputs_raw: Optional = None
#     outputs: Optional = None
#     expected: Optional = None
#     diff: Optional = None

#     model = load_description(
#         model_rdf, weights_priority_order=None if weight_format is None else [weight_format]
#     )
#     if not isinstance(model, Model):
#         raise ValueError(f"Not a bioimageio.model: {model_rdf}")

#     prediction_pipeline = create_prediction_pipeline(
#         bioimageio_model=model, devices=devices, weight_format=weight_format
#     )
#     inputs = [
#         xr.DataArray(load_array(str(in_path)), dims=input_spec.axes)
#         for in_path, input_spec in zip(model.test_inputs, model.inputs)
#     ]
#     input_dict = {input_spec.name: input for input_spec, input in zip(model.inputs, inputs)}

#     # keep track of the non-processed inputs
#     inputs_raw = [deepcopy(input) for input in inputs]

#     computed_measures = {}

#     prediction_pipeline.apply_preprocessing(input_dict, computed_measures)
#     inputs_processed = list(input_dict.values())
#     outputs_raw = prediction_pipeline.predict(*inputs_processed)
#     output_dict = {output_spec.name: deepcopy(output) for output_spec, output in zip(model.outputs, outputs_raw)}
#     prediction_pipeline.apply_postprocessing(output_dict, computed_measures)
#     outputs = list(output_dict.values())

#     if isinstance(outputs, (np.ndarray, xr.DataArray)):
#         outputs = [outputs]

#     expected = [
#         xr.DataArray(load_array(str(out_path)), dims=output_spec.axes)
#         for out_path, output_spec in zip(model.test_outputs, model.outputs)
#     ]
#     if len(outputs) != len(expected):
#         error = f"Number of outputs and number of expected outputs disagree: {len(outputs)} != {len(expected)}"
#         print(error)
#     else:
#         diff = []
#         for res, exp in zip(outputs, expected):
#             diff.append(res - exp)

#     return {
#         "inputs": inputs_raw,
#         "inputs_processed": inputs_processed,
#         "outputs_raw": outputs_raw,
#         "outputs": outputs,
#         "expected": expected,
#         "diff": diff,
#     }
