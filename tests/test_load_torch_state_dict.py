from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch is not installed", allow_module_level=True)

try:
    import safetensors.torch
except ImportError:
    pytest.skip("safetensors.torch is not installed", allow_module_level=True)

from bioimageio.core.backends.pytorch_backend import load_torch_state_dict
from bioimageio.spec.common import BytesReader, ZipPath


def _make_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))


@pytest.fixture(scope="module")
def reference_model() -> torch.nn.Module:
    _ = torch.manual_seed(0)
    return _make_model()


@pytest.fixture(scope="module")
def weights_path(
    tmp_path_factory: pytest.TempPathFactory, reference_model: torch.nn.Module
) -> Path:
    path = tmp_path_factory.mktemp("safetensors") / "weights.safetensors"
    safetensors.torch.save_file(reference_model.state_dict(), str(path))
    return path


def _assert_same_weights(a: torch.nn.Module, b: torch.nn.Module):
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k


def _fresh_model(reference_model: torch.nn.Module) -> torch.nn.Module:
    _ = torch.manual_seed(1)
    model = _make_model()
    with pytest.raises(AssertionError):
        _assert_same_weights(model, reference_model)

    return model


def test_load_safetensors_from_path(
    weights_path: Path, reference_model: torch.nn.Module
):
    model = load_torch_state_dict(
        _fresh_model(reference_model), weights_path, [torch.device("cpu")]
    )
    _assert_same_weights(model, reference_model)


def test_load_safetensors_from_zip_path(
    weights_path: Path, reference_model: torch.nn.Module, tmp_path: Path
):
    zip_file = tmp_path / "weights.zip"
    with ZipFile(zip_file, "w") as zf:
        zf.write(weights_path, arcname="weights.safetensors")

    model = load_torch_state_dict(
        _fresh_model(reference_model),
        ZipPath(zip_file, "weights.safetensors"),
        [torch.device("cpu")],
    )
    _assert_same_weights(model, reference_model)


def test_load_safetensors_from_bytes_reader(
    weights_path: Path, reference_model: torch.nn.Module
):
    reader = BytesReader(
        BytesIO(weights_path.read_bytes()),
        sha256=None,
        suffix=".safetensors",
        original_file_name=weights_path.name,
        original_root=weights_path.parent,
        is_zipfile=False,
    )
    model = load_torch_state_dict(
        _fresh_model(reference_model), reader, [torch.device("cpu")]
    )
    _assert_same_weights(model, reference_model)
