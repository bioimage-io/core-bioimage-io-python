# core-bioimage-io-python

Python specific core utilities for running models in the [BioImage Model Zoo](https://bioimage.io).

## Installation

### Via Conda

The `bioimageio.core` package can be installed from conda-forge via

```console
conda install -c conda-forge bioimageio.core
```

If you do not install any additional deep learning libraries, you will only be able to use general convenience
functionality, but not any functionality for model prediction.
To install additional deep learning libraries use:

* Pytorch/Torchscript:

  CPU installation (if you don't have an nvidia graphics card):

  ```console
  conda install -c pytorch -c conda-forge bioimageio.core pytorch torchvision cpuonly
  ```

  GPU installation (for cuda 11.6, please choose the appropriate cuda version for your system):

  ```console
  conda install -c pytorch -c nvidia -c conda-forge bioimageio.core pytorch torchvision pytorch-cuda=11.6
  ```

  Note that the pytorch installation instructions may change in the future. For the latest instructions please refer to [pytorch.org](https://pytorch.org/).

* Tensorflow

  Currently only CPU version supported

  ```console
  conda install -c conda-forge bioimageio.core tensorflow
  ```

* ONNXRuntime

  Currently only cpu version supported

  ```console
  conda install -c conda-forge bioimageio.core onnxruntime
  ```

### Via pip

The package is also available via pip:

```console
pip install bioimageio.core
```

### Set up Development Environment

To set up a development conda environment run the following commands:

```console
conda env create -f dev/environment-base.yaml
conda activate bio-core-dev
pip install -e . --no-deps
```

There are different environment files that only install tensorflow or pytorch as dependencies available.

## 💻 Command Line

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

## From python

`bioimageio.core` is a python library that implements loading models, running prediction with them and more.
To get an overview of this functionality, check out the example notebooks:

* [example/model_usage](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_usage.ipynb) for how to load models and run prediction with them
* [example/model_creation](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_creation.ipynb) for how to create bioimage.io compatible model packages
* [example/dataset_statistics_demo](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/dataset_statistics_demo.ipynb) for how to use the dataset statistics for advanced pre-and-postprocessing

## Model Specification

The model specification and its validation tools can be found at <https://github.com/bioimage-io/spec-bioimage-io>.

## Changelog

### 0.5.10

* [Fix critical bug in predict with tiling](https://github.com/bioimage-io/core-bioimage-io-python/pull/359)
