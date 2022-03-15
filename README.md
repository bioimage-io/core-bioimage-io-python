# core-bioimage-io-python

Python specific core utilities for running models in the [BioImage Model Zoo](https://bioimage.io).

## Installation

### Via Conda

The `bioimageio.core` package can be installed from conda-forge via
```
conda install -c conda-forge bioimageio.core
```
if you don't install any additional deep learning libraries, you will only be able to use general convenience functionality, but not any functionality for model prediction.
To install additional deep learning libraries use:

* Pytorch/Torchscript:
  ```bash
  # cpu installation (if you don't have an nvidia graphics card)
  conda install -c pytorch -c conda-forge bioimageio.core pytorch torchvision cpuonly

  # gpu installation
  conda install -c pytorch -c conda-forge bioimageio.core pytorch torchvision cudatoolkit
  ```

* Tensorflow
  ```bash
  # currently only cpu version supported
  conda install -c conda-forge bioimageio.core tensorflow
  ```

* ONNXRuntime
  ```bash
  # currently only cpu version supported
  conda install -c conda-forge bioimageio.core onnxruntime
  ```
  
### Via pip

The package is also available via pip:
```
pip install bioimageio.core
```

### Set up Development Environment

To set up a development conda environment run the following commands:
```
conda env create -f dev/environment-base.yaml
conda activate bio-core-dev
pip install -e . --no-deps
```

There are different environment files that only install tensorflow or pytorch as dependencies available.

## Command Line

`bioimageio.core` installs a command line interface for testing models and other functionality. You can list all the available commands via:
```
bioimageio
```

Check that a model adheres to the model spec:
```
bioimageio validate <MODEL>
```

Test a model (including prediction for the test input):
```
bioimageio test-model <MODEL>
```

Run prediction for an image stored on disc:
```
bioimageio predict-image -m <MODEL> -i <INPUT> -o <OUTPUT>
```

Run prediction for multiple images stored on disc:
```
bioimagei predict-images -m <MODEL> -i <INPUT_PATTERN> - o <OUTPUT_FOLDER>
```
`<INPUT_PATTERN>` is a `glob` pattern to select the desired images, e.g. `/path/to/my/images/*.tif`.


## From python

`bioimageio.core` can be used as a python library. See the notebook [example/bioimageio-core-usage.ipynb](https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/bioimageio-core-usage.ipynb) for usage examples.

## Model Specification

The model specification and its validation tools can be found at https://github.com/bioimage-io/spec-bioimage-io.
