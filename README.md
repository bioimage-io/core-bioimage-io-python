# python-bioimage-io core

Python specific core utilities for working with the BioimageIO model zoo.

## Installation

### Via Conda

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
  conda install -c conda-forge -c ilastik-forge bioimageio.core onnxruntime
  ```

### Set up Development Environment

To set up a development conda enveironment run the following commands:
```
conda env create -f dev/environment-base.yaml
conda activate bio-core-dev
pip install -e . --no-deps
```

## Command Line

Test a model:
```
bioimageio-test -m <MODEL>
```

Run prediction:
```
bioimageio-predict -m <MODEL> -i <INPUT> -o <OUTPUT>
```

This is subject to change, see https://github.com/bioimage-io/python-bioimage-io/issues/87.


## Running network predictions:

TODO

## Model Specification

The model specification and its validation tools can be found at https://github.com/bioimage-io/spec-bioimage-io.
