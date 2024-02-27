import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from bioimageio.core import load_description


def run_subprocess(commands: Sequence[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", **kwargs)


def test_validate_model(unet2d_nuclei_broad_model: Path):
    ret = run_subprocess(["bioimageio", "validate", unet2d_nuclei_broad_model])
    assert ret.returncode == 0, ret.stdout


def test_cli_package(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    out_path = tmp_path / "model.zip"
    ret = run_subprocess(["bioimageio", "package", unet2d_nuclei_broad_model, str(out_path)])
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()


def test_cli_package_wo_cache(unet2d_nuclei_broad_model: Path):
    env = os.environ.copy()
    env["BIOIMAGEIO_USE_CACHE"] = "false"
    ret = run_subprocess(["bioimageio", "package", unet2d_nuclei_broad_model], env=env)
    assert ret.returncode == 0, ret.stdout


def test_cli_test_model(unet2d_nuclei_broad_model: Path):
    ret = run_subprocess(["bioimageio", "test-model", unet2d_nuclei_broad_model])
    assert ret.returncode == 0, ret.stdout


def test_cli_test_model_fail(stardist_wrong_shape: Path):
    ret = run_subprocess(["bioimageio", "test-model", stardist_wrong_shape])
    assert ret.returncode == 1


def test_cli_test_model_with_weight_format(unet2d_nuclei_broad_model: Path):
    ret = run_subprocess(
        ["bioimageio", "test-model", unet2d_nuclei_broad_model, "--weight-format", "pytorch_state_dict"]
    )
    assert ret.returncode == 0, ret.stdout


def test_cli_test_resource(unet2d_nuclei_broad_model: Path):
    ret = run_subprocess(["bioimageio", "test-resource", unet2d_nuclei_broad_model])
    assert ret.returncode == 0, ret.stdout


def test_cli_test_resource_with_weight_format(unet2d_nuclei_broad_model: Path):
    ret = run_subprocess(
        ["bioimageio", "test-resource", unet2d_nuclei_broad_model, "--weight-format", "pytorch_state_dict"]
    )
    assert ret.returncode == 0, ret.stdout


def _test_cli_predict_image(model: Path, tmp_path: Path, extra_cmd_args: Optional[List[str]] = None):
    spec = load_description(model)
    in_path = spec.test_inputs[0]

    out_path = tmp_path.with_suffix(".npy")
    cmd = ["bioimageio", "predict-image", model, "--input", str(in_path), "--output", str(out_path)]
    if extra_cmd_args is not None:
        cmd.extend(extra_cmd_args)
    ret = run_subprocess(cmd)
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()


def test_cli_predict_image(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_image(unet2d_nuclei_broad_model, tmp_path)


def test_cli_predict_image_with_weight_format(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_image(unet2d_nuclei_broad_model, tmp_path, ["--weight-format", "pytorch_state_dict"])


def _test_cli_predict_images(model: Path, tmp_path: Path, extra_cmd_args: Optional[List[str]] = None):
    n_images = 3
    shape = (1, 1, 128, 128)
    expected_shape = (1, 1, 128, 128)

    in_folder = tmp_path / "inputs"
    in_folder.mkdir()
    out_folder = tmp_path / "outputs"
    out_folder.mkdir()

    expected_outputs: List[Path] = []
    for i in range(n_images):
        path = in_folder / f"im-{i}.npy"
        im = np.random.randint(0, 255, size=shape).astype("uint8")
        np.save(path, im)
        expected_outputs.append(out_folder / f"im-{i}.npy")

    input_pattern = str(in_folder / "*.npy")
    cmd = ["bioimageio", "predict-images", str(model), input_pattern, str(out_folder)]
    if extra_cmd_args is not None:
        cmd.extend(extra_cmd_args)
    ret = run_subprocess(cmd)
    assert ret.returncode == 0, ret.stdout

    for out_path in expected_outputs:
        assert out_path.exists()
        assert np.load(out_path).shape == expected_shape


def test_cli_predict_images(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_images(unet2d_nuclei_broad_model, tmp_path)


def test_cli_predict_images_with_weight_format(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    _test_cli_predict_images(unet2d_nuclei_broad_model, tmp_path, ["--weight-format", "pytorch_state_dict"])


def test_torch_to_torchscript(unet2d_nuclei_broad_model: Path, tmp_path: Path):
    out_path = tmp_path.with_suffix(".pt")
    ret = run_subprocess(
        ["bioimageio", "convert-torch-weights-to-torchscript", str(unet2d_nuclei_broad_model), str(out_path)]
    )
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()


def test_torch_to_onnx(convert_to_onnx: Path, tmp_path: Path):
    out_path = tmp_path.with_suffix(".onnx")
    ret = run_subprocess(["bioimageio", "convert-torch-weights-to-onnx", str(convert_to_onnx), str(out_path)])
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()


def test_keras_to_tf(unet2d_keras: Path, tmp_path: Path):
    out_path = tmp_path / "weights.zip"
    ret = run_subprocess(["bioimageio", "convert-keras-weights-to-tensorflow", str(unet2d_keras), str(out_path)])
    assert ret.returncode == 0, ret.stdout
    assert out_path.exists()
