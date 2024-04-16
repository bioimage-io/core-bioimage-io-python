def test_save_bioimageio_package(unet2d_nuclei_broad_model: str):
    from bioimageio.spec._package import save_bioimageio_package

    _ = save_bioimageio_package(
        unet2d_nuclei_broad_model,
        weights_priority_order=(
            None
            if weights_priority_order is None
            else [wpo.name for wpo in weights_priority_order]
        ),
    )
