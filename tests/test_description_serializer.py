def test_description_serializer_roundtrip(unet2d_nuclei_broad_model: str):
    from bioimageio.core._description_serializer import DescriptionSerializer
    from bioimageio.spec import load_model_description

    rd1 = load_model_description(unet2d_nuclei_broad_model, perform_io_checks=False)

    serialized = DescriptionSerializer.serialize_to_string(rd1)
    assert len(serialized) > 2083

    rd2 = DescriptionSerializer.deserialize_from_string(serialized)
    assert rd2.name == rd1.name
    assert rd2.type == rd1.type
    assert rd2.format_version == rd1.format_version
    for k, v in rd1:
        assert str(getattr(rd2, k)) == str(v), (k, v, getattr(rd2, k))
