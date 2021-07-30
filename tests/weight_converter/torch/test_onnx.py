import os


# TODO only run this test if we have pytorch
# TODO clean up
def test_torchscript_converter(unet2d_nuclei_broad_model):
    from bioimageio.core.weight_converter.torch.onnx import convert_weights_to_onnx
    out_path = "./weights.onnx"
    convert_weights_to_onnx(unet2d_nuclei_broad_model, out_path)
    assert os.path.exists(out_path)
