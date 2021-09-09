import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from marshmallow import missing

from bioimageio.spec.model import raw_nodes


def test_export_package(unet2d_nuclei_broad_model):
    from bioimageio.core import export_resource_package, load_raw_resource_description

    package_path = export_resource_package(unet2d_nuclei_broad_model, weights_priority_order=["onnx"])
    assert isinstance(package_path, Path), package_path
    assert package_path.exists(), package_path

    raw_model = load_raw_resource_description(package_path)
    assert isinstance(raw_model, raw_nodes.Model)


def test_package_with_folder(unet2d_nuclei_broad_model):
    from bioimageio.core import export_resource_package, load_raw_resource_description

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # extract package (to not cache to BIOIMAGEIO_CACHE)
        package_folder = tmp_dir / "package"
        with ZipFile(unet2d_nuclei_broad_model) as zf:
            zf.extractall(package_folder)

        # load package
        model = load_raw_resource_description(package_folder / "rdf.yaml")
        assert isinstance(model, raw_nodes.Model)

        # alter package to have its documentation in a nested folder
        doc = model.documentation
        assert doc is not missing
        assert not doc.is_absolute()
        new_doc = Path("nested") / "folder" / doc
        (package_folder / new_doc).parent.mkdir(parents=True)
        shutil.move(package_folder / doc, package_folder / new_doc)
        model.documentation = new_doc

        # export altered package
        altered_package = tmp_dir / "altered_package.zip"
        altered_package = export_resource_package(model, output_path=altered_package, weights_priority_order=["onnx"])

        # extract altered package (to not cache to BIOIMAGEIO_CACHE)
        altered_package_folder = tmp_dir / "altered_package"
        with ZipFile(altered_package) as zf:
            zf.extractall(altered_package_folder)

        # load altered package
        reloaded_model = load_raw_resource_description(altered_package_folder / "rdf.yaml")
        assert isinstance(reloaded_model, raw_nodes.Model)
        assert reloaded_model.documentation == new_doc
        assert (altered_package_folder / reloaded_model.documentation).exists()
