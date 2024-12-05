import subprocess
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    assert_never,
)

from loguru import logger
from typing_extensions import get_args

from bioimageio.spec import (
    BioimageioCondaEnv,
    ValidationSummary,
    get_conda_env,
    load_description,
)
from bioimageio.spec._internal.io import is_yaml_value
from bioimageio.spec._internal.io_utils import write_yaml
from bioimageio.spec.common import PermissiveFileSource
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import WeightsFormat


def default_run_command(args: Sequence[str]):
    logger.info("running '{}'...", " ".join(args))
    _ = subprocess.run(args, shell=True, text=True, check=True)


def test_description_in_conda_env(
    source: PermissiveFileSource,
    *,
    weight_format: Optional[WeightsFormat] = None,
    conda_env: Optional[BioimageioCondaEnv] = None,
    devices: Optional[List[str]] = None,
    absolute_tolerance: float = 1.5e-4,
    relative_tolerance: float = 1e-4,
    determinism: Literal["seed_only", "full"] = "seed_only",
    run_command: Callable[[Sequence[str]], None] = default_run_command,
) -> ValidationSummary:
    """Run test_model in a dedicated conda env

    Args:
        source: Path or URL to model description.
        weight_format: Weight format to test.
            Default: All weight formats present in **source**.
        conda_env: conda environment including bioimageio.core dependency.
            Default: Use `bioimageio.spec.get_conda_env` to obtain a model weight
            specific conda environment.
        devices: Devices to test with, e.g. 'cpu', 'cuda'.
            Default (may be weight format dependent): ['cuda'] if available, ['cpu'] otherwise.
        absolute_tolerance: Maximum absolute tolerance of reproduced output tensors.
        relative_tolerance: Maximum relative tolerance of reproduced output tensors.
        determinism: Modes to improve reproducibility of test outputs.
        run_command: Function to execute terminal commands.
    """

    try:
        run_command(["which", "conda"])
    except Exception as e:
        raise RuntimeError("Conda not available") from e

    descr = load_description(source)
    if not isinstance(descr, (v0_4.ModelDescr, v0_5.ModelDescr)):
        raise NotImplementedError("Not yet implemented for non-model resources")

    if weight_format is None:
        all_present_wfs = [
            wf for wf in get_args(WeightsFormat) if getattr(descr.weights, wf)
        ]
        ignore_wfs = [wf for wf in all_present_wfs if wf in ["tensorflow_js"]]
        logger.info(
            "Found weight formats {}. Start testing all{}...",
            all_present_wfs,
            f" (except: {', '.join(ignore_wfs)}) " if ignore_wfs else "",
        )
        summary = test_description_in_env(
            source,
            weight_format=all_present_wfs[0],
            devices=devices,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            determinism=determinism,
        )
        for wf in all_present_wfs[1:]:
            additional_summary = test_description_in_env(
                source,
                weight_format=all_present_wfs[0],
                devices=devices,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                determinism=determinism,
            )
            for d in additional_summary.details:
                # TODO: filter reduntant details; group details
                summary.add_detail(d)
        return summary

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

    # remove name as we crate a name based on the env description hash value
    conda_env.name = None

    dumped_env = conda_env.model_dump(mode="json", exclude_none=True)
    if not is_yaml_value(dumped_env):
        raise ValueError(f"Failed to dump conda env to valid YAML {conda_env}")

    env_io = StringIO()
    write_yaml(dumped_env, file=env_io)
    encoded_env = env_io.getvalue().encode()
    env_name = sha256(encoded_env).hexdigest()

    with TemporaryDirectory() as _d:
        folder = Path(_d)
        try:
            run_command(["conda", "activate", env_name])
        except Exception:
            path = folder / "env.yaml"
            _ = path.write_bytes(encoded_env)

            run_command(
                ["conda", "env", "create", "--file", str(path), "--name", env_name]
            )
            run_command(["conda", "activate", env_name])

        summary_path = folder / "summary.json"
        run_command(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "bioimageio",
                "test",
                str(source),
                "--summary-path",
                str(summary_path),
            ]
        )
        return ValidationSummary.model_validate_json(summary_path.read_bytes())
