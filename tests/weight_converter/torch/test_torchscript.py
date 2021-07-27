from bioimageio.spec import export_package


# TODO only run this test if we have pytorch
def test_torchscript_converter(unet2d_nuclei_broad_model_url):
    spec_path = export_package(unet2d_nuclei_broad_model_url)

