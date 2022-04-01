def test_replace_values():
    from bioimageio.core.build_spec.util import replace_values

    input_dict = {"x": 1, "y": 2, "z": 3, "xyz": {"x": 4, "y": 5}}

    new_dict = replace_values(input_dict, {"x": 42})
    assert new_dict["x"] == 42
    assert new_dict["xyz"]["x"] == 42
    assert new_dict["y"] == 2
    assert new_dict["z"] == 3

    new_dict = replace_values(input_dict, {"x": 42, "y": 66})
    assert new_dict["x"] == 42
    assert new_dict["xyz"]["x"] == 42
    assert new_dict["xyz"]["y"] == 66
    assert new_dict["y"] == 66
    assert new_dict["z"] == 3
