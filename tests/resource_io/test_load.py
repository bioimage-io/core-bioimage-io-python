def test_load_v0_1(unet2d_nuclei_broad_v0_1_path):
    from bioimageio.core.resource_io import load_resource_description, nodes

    model = load_resource_description(unet2d_nuclei_broad_v0_1_path)
    assert isinstance(model, nodes.Model)
    assert model.documentation.exists()
