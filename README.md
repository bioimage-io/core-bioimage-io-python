# core-bioimage-io-python

Python specific core utilities for bioimage.io resources (in particular models).

## Installation

### Via Mamba/Conda

The `bioimageio.core` package can be installed from conda-forge via

```console
mamba install -c conda-forge bioimageio.core
```

If you do not install any additional deep learning libraries, you will only be able to use general convenience
functionality, but not any functionality for model prediction.
To install additional deep learning libraries use:

* Pytorch/Torchscript:

  CPU installation (if you don't have an nvidia graphics card):

  ```console
  mamba install -c pytorch -c conda-forge bioimageio.core pytorch torchvision cpuonly
  ```

  GPU installation (for cuda 11.6, please choose the appropriate cuda version for your system):

  ```console
  mamba install -c pytorch -c nvidia -c conda-forge bioimageio.core pytorch torchvision pytorch-cuda=11.8
  ```

  Note that the pytorch installation instructions may change in the future. For the latest instructions please refer to [pytorch.org](https://pytorch.org/).

* Tensorflow

  Currently only CPU version supported

  ```console
  mamba install -c conda-forge bioimageio.core tensorflow
  ```

* ONNXRuntime

  Currently only cpu version supported

  ```console
  mamba install -c conda-forge bioimageio.core onnxruntime
  ```

### Via pip

The package is also available via pip
(e.g. with recommended extras `onnx` and `pytorch`):

```console
pip install bioimageio.core[onnx,pytorch]
```

### Set up Development Environment

To set up a development conda environment run the following commands:

```console
mamba env create -f dev/env.yaml
mamba activate core
pip install -e . --no-deps
```

There are different environment files available that only install tensorflow or pytorch as dependencies.

## 💻 Use the Command Line Interface

`bioimageio.core` installs a command line interface (CLI) for testing models and other functionality.
You can list all the available commands via:

```console
bioimageio
```

Check that a model adheres to the model spec:

```console
bioimageio validate <MODEL>
```

Test a model (including prediction for the test input):

```console
bioimageio test-model <MODEL>
```

Run prediction for an image stored on disc:

```console
bioimageio predict-image <MODEL> --inputs <INPUT> --outputs <OUTPUT>
```

Run prediction for multiple images stored on disc:

```console
bioimagei predict-images -m <MODEL> -i <INPUT_PATTERN> - o <OUTPUT_FOLDER>
```

`<INPUT_PATTERN>` is a `glob` pattern to select the desired images, e.g. `/path/to/my/images/*.tif`.

## 🐍 Use in Python

`bioimageio.core` is a python package that implements prediction with bioimageio models
including standardized pre- and postprocessing operations.
These models are described by---and can be loaded with---the bioimageio.spec package.

In addition bioimageio.core provides functionality to convert model weight formats.

To get an overview of this functionality, check out these example notebooks:

* [model creation/loading with bioimageio.spec](https://github.com/bioimage-io/spec-bioimage-io/blob/main/example_use/load_model_and_create_your_own.ipynb)

and the [developer documentation](https://bioimage-io.github.io/core-bioimage-io-python/bioimageio/core.html).

## Model Specification

The model specification and its validation tools can be found at <https://github.com/bioimage-io/spec-bioimage-io>.

## Changelog

### 0.6.5

* improve adapter error messages

### 0.6.4

* add `bioimageio validate-format` command
* improve error messages and display of command results

### 0.6.3

* Fix [#386](https://github.com/bioimage-io/core-bioimage-io-python/issues/386)
* (in model inference testing) stop assuming model inputs are tileable

### 0.6.2

* Fix [#384](https://github.com/bioimage-io/core-bioimage-io-python/issues/384)

### 0.6.1

* Fix [#378](https://github.com/bioimage-io/core-bioimage-io-python/pull/378) (with [#379](https://github.com/bioimage-io/core-bioimage-io-python/pull/379))*

### 0.6.0

* add compatibility with new bioimageio.spec 0.5 (0.5.2post1)
* improve interfaces

### 0.5.10

* [Fix critical bug in predict with tiling](https://github.com/bioimage-io/core-bioimage-io-python/pull/359)
