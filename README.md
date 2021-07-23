# python-bioimage-io core

Python specific core utilities for working with the BioimageIO model zoo.

## Installation

The `bioimageio.core` package supports various back-ends for running BioimageIO networks:

* Pytorch/Torchscript:
  ```bash
  # cpu installation (if you don't have an nvidia graphics card)
  conda install -c pytorch -c conda-forge -c ilastik-forge bioimageio.core pytorch torchvision cpuonly

  # gpu installation
  conda install -c pytorch -c conda-forge -c ilastik-forge bioimageio.core pytorch torchvision cudatoolkit
  ```

* Tensorflow
  ```bash
  # currently only cpu version supported
  conda install -c conda-forge -c ilastik-forge bioimageio.core tensorflow
  ```

* ONNXRuntime
  ```bash
  # currently only cpu version supported
  install -c conda-forge -c ilastik-forge bioimageio.core onnxruntime
  ```


## Running network predictions:

TODO

## Model Specification

The model specification and its validation tools can be found at https://github.com/bioimage-io/spec-bioimage-io.
