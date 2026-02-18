## Via Conda

The `bioimageio.core` package can be installed from conda-forge via

```console
conda install -c conda-forge bioimageio.core
```

If you do not install any additional deep learning (DL) libraries, you will only be able to use general convenience
functionality, but model inference will be unavailable.
To install additional deep learning libraries add `pytorch`, `onnxruntime`, `keras` or `tensorflow`.

Deeplearning frameworks to consider installing alongside `bioimageio.core`:

- [Pytorch/Torchscript](https://pytorch.org/get-started/locally/)
- [TensorFlow](https://www.tensorflow.org/install)
- [ONNXRuntime](https://onnxruntime.ai/docs/install/#python-installs)

Example for installing bioimageio.core via conda with additional DL frameworks:

```console
conda install -c conda-forge bioimageio.core pytorch torchvision onnxruntime
```

## Via pip

The package is also available via pip
(e.g. with recommended extras `onnx` and `pytorch`):

```console
pip install "bioimageio.core[pytorch,onnx]"
```
