def test_save_bioimageio_package(unet2d_nuclei_broad_model: str):
    from bioimageio.spec._package import save_bioimageio_package

    _ = save_bioimageio_package(
        unet2d_nuclei_broad_model,
        weights_priority_order=("pytorch_state_dict",),
    )
