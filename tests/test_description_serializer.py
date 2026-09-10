def test_description_serializer_roundtrip(unet2d_nuclei_broad_model: str):
    from bioimageio.core._description_serializer import DescriptionSerializer
    from bioimageio.spec import load_model_description

    rd1 = load_model_description(unet2d_nuclei_broad_model, perform_io_checks=False)

    serialized1, hash1 = DescriptionSerializer.serialize_to_string_and_hash(rd1)
    assert len(serialized1) > 2083

    rd2 = DescriptionSerializer.deserialize_from_string(serialized1)
    assert rd2.name == rd1.name
    assert rd2.type == rd1.type
    assert rd2.format_version == rd1.format_version
    for k, v in rd1:
        assert str(getattr(rd2, k)) == str(v), (k, v, getattr(rd2, k))

    serialized2, hash2 = DescriptionSerializer.serialize_to_string_and_hash(rd2)
    assert serialized2 == serialized1
    assert hash2 == hash1
