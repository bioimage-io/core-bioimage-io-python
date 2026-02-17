import subprocess
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pytest
from pydantic import FilePath

from bioimageio.spec import load_description, settings


def run_subprocess(
    commands: Sequence[str], **kwargs: Any
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        **kwargs,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["add-weights", "unet2d_nuclei_broad_model", "tmp_path"],
        [
            "package",
            "unet2d_nuclei_broad_model",
            "output.zip",
            "--weight_format",
            "pytorch_state_dict",
        ],
        ["package", "unet2d_nuclei_broad_model", "output.zip"],
        ["predict", "--example", "unet2d_nuclei_broad_model"],
        [
            "test",
            "unet2d_nuclei_broad_model",
            "--weight_format",
            "pytorch_state_dict",
        ],
        ["test", "unet2d_nuclei_broad_model"],
        ["update-format", "unet2d_nuclei_broad_model_old"],
        ["update-hashes", "unet2d_nuclei_broad_model_old", "--output=stdout"],
        ["update-hashes", "unet2d_nuclei_broad_model_old"],
    ],
)
def test_cli(
    args: List[str],
    unet2d_nuclei_broad_model: str,
    unet2d_nuclei_broad_model_old: str,
    tmp_path: Path,
):
    resolved_args = [
        (
            unet2d_nuclei_broad_model
            if arg == "unet2d_nuclei_broad_model"
            else (
                unet2d_nuclei_broad_model_old
                if arg == "unet2d_nuclei_broad_model_old"
                else (
                    arg.replace("tmp_path", str(tmp_path)) if "tmp_path" in arg else arg
                )
            )
        )
        for arg in args
    ]
    ret = run_subprocess(["bioimageio", *resolved_args])
    assert ret.returncode == 0, ret.stdout


def test_empty_cache(tmp_path: Path, unet2d_nuclei_broad_model: str):
    from bioimageio.spec.utils import empty_cache

    origingal_cache_path = settings.cache_path
    try:
        settings.cache_path = tmp_path / "cache"
        assert not settings.cache_path.exists()
        _ = load_description(unet2d_nuclei_broad_model, perform_io_checks=False)
        assert (
            len([fn for fn in settings.cache_path.iterdir() if fn.suffix != ".lock"])
            == 1
        )
        empty_cache()
        assert (
            len([fn for fn in settings.cache_path.iterdir() if fn.suffix != ".lock"])
            == 0
        )
    finally:
        settings.cache_path = origingal_cache_path


@pytest.mark.parametrize("args", [["test", "stardist_wrong_shape"]])
def test_cli_fails(args: List[str], stardist_wrong_shape: FilePath):
    resolved_args = [
        str(stardist_wrong_shape) if arg == "stardist_wrong_shape" else arg
        for arg in args
    ]
    ret = run_subprocess(["bioimageio", *resolved_args])
    assert ret.returncode == 1, ret.stdout


def _test_cli_predict_single(
    model_source: str, tmp_path: Path, extra_cmd_args: Sequence[str] = ()
):
    from bioimageio.spec import load_model_description

    model = load_model_description(model_source, format_version="latest")
    assert model.inputs[0].test_tensor is not None
    in_path = tmp_path / "in.npy"
    _ = in_path.write_bytes(model.inputs[0].test_tensor.get_reader().read())
    out_path = tmp_path / "out.npy"
    cmd = [
        "bioimageio",
        "predict",
        str(model_source),
        "--inputs",
        str(in_path),
        "--outputs",
        str(out_path),
    ] + list(extra_cmd_args)
    ret = run_subprocess(cmd)
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()


def test_cli_predict_single(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_single(str(unet2d_nuclei_broad_model), tmp_path)


def test_cli_predict_single_with_weight_format(
    unet2d_nuclei_broad_model: Path, tmp_path: Path
):
    _test_cli_predict_single(
        str(unet2d_nuclei_broad_model),
        tmp_path,
        ["--weight-format", "pytorch_state_dict"],
    )


def _test_cli_predict_multiple(
    model_source: str, tmp_path: Path, extra_cmd_args: Sequence[str] = ()
):
    n_images = 3
    shape = (1, 1, 128, 128)
    expected_shape = (1, 1, 128, 128)

    in_folder = tmp_path / "inputs"
    in_folder.mkdir()
    out_folder = tmp_path / "outputs"
    out_folder.mkdir()
    out_file_pattern = "im-{sample_id}.npy"
    inputs: List[str] = []
    expected_outputs: List[Path] = []
    for i in range(n_images):
        input_path = in_folder / f"im-{i}.npy"
        im = np.random.randint(0, 255, size=shape).astype("uint8")
        np.save(input_path, im)
        inputs.extend(["--inputs", str(input_path)])
        expected_outputs.append(out_folder / out_file_pattern.format(sample_id=i))

    cmd = [
        "bioimageio",
        "predict",
        model_source,
        *inputs,
        "--outputs",
        str(out_folder / out_file_pattern),
    ] + list(extra_cmd_args)
    ret = run_subprocess(cmd)
    assert ret.returncode == 0, ret.stdout

    for out_path in expected_outputs:
        assert out_path.exists()
        assert np.load(out_path).shape == expected_shape


def test_cli_predict_multiple(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_multiple(str(unet2d_nuclei_broad_model), tmp_path)


def test_cli_predict_multiple_with_weight_format(
    unet2d_nuclei_broad_model: Path, tmp_path: Path
):
    _test_cli_predict_multiple(
        str(unet2d_nuclei_broad_model),
        tmp_path,
        ["--weight-format", "pytorch_state_dict"],
    )
