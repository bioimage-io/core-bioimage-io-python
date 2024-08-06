![License](https://img.shields.io/github/license/bioimage-io/core-bioimage-io-python.svg)
[![PyPI](https://img.shields.io/pypi/v/bioimageio-core.svg?style=popout)](https://pypi.org/project/bioimageio.core/)
[![conda-version](https://anaconda.org/conda-forge/bioimageio.core/badges/version.svg)](https://anaconda.org/conda-forge/bioimageio.core/)
[![downloads](https://static.pepy.tech/badge/bioimageio.core)](https://pepy.tech/project/bioimageio.core)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/bioimageio.core.svg?label=conda-forge)](https://anaconda.org/conda-forge/bioimageio.core/)
![code style](https://img.shields.io/badge/code%20style-black-000000.svg)

# core-bioimage-io-python

Python specific core utilities for bioimage.io resources (in particular models).

## Get started

To get started we recommend installing bioimageio.core with conda together with a deep
learning framework, e.g. pytorch, and run a few `bioimageio` commands to see what
bioimage.core offers.

1. install with conda (for more details on conda environments, [checkout the conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

```console
install -c conda-forge bioimageio.core pytorch
```

1. test a model

```console
bioimageio test powerful-chipmunk

testing powerful-chipmunk...
2024-07-24 17:10:37.470 | INFO     | bioimageio.spec._internal.io_utils:open_bioimageio_yaml:112 - loading powerful-chipmunk from https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/powerful-chipmunk/1/files/rdf.yaml
Updating data from 'https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/powerful-chipmunk/1/files/rdf.yaml' to file 'C:\Users\fbeut\AppData\Local\bioimageio\bioimageio\Cache\d968304289dc978b9221e813dc757a3a-rdf.yaml'.
100%|#####################################| 2.92k/2.92k [00:00<00:00, 1.53MB/s]
computing SHA256 of 1e659a86d8dd8a7c6cfb3315f4447f5d-weights.pt (result: 3bd9c518c8473f1e35abb7624f82f3aa92f1015e66fb1f6a9d08444e1f2f5698): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 884/884 [00:00<00:00, 1006.20it/s]
computing SHA256 of 97a83ece802cfc5ba362aa76b5f77c3a-weights-torchscript.pt (result: 4e568fd81c0ffa06ce13061327c3f673e1bac808891135badd3b0fcdacee086b): 100%|██████████████████████████████████████████████████████████████████████████████████████| 885/885 [00:00<00:00, 1229.39it/s]
2024-07-24 17:10:44.596 | INFO     | bioimageio.core._resource_tests:_test_model_inference:130 - starting 'Reproduce test outputs from test inputs'
2024-07-24 17:11:00.136 | INFO     | bioimageio.core._resource_tests:_test_model_inference:130 - starting 'Reproduce test outputs from test inputs'


  ✔️                 bioimageio validation passed
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  source            https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/powerful-chipmunk/1/files/rdf.yaml
  format version    model 0.4.10
  bioimageio.spec   0.5.3post4
  bioimageio.core   0.6.8



  ❓   location                                     detail
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✔️                                                 initialized ModelDescr to describe model 0.4.10

  ✔️                                                 bioimageio.spec format validation model 0.4.10
  🔍   context.perform_io_checks                    True
  🔍   context.root                                 https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/powerful-chipmunk/1/files
  🔍   context.known_files.weights.pt               3bd9c518c8473f1e35abb7624f82f3aa92f1015e66fb1f6a9d08444e1f2f5698
  🔍   context.known_files.weights-torchscript.pt   4e568fd81c0ffa06ce13061327c3f673e1bac808891135badd3b0fcdacee086b
  🔍   context.warning_level                        error

  ✔️                                                 Reproduce test outputs from test inputs

  ✔️                                                 Reproduce test outputs from test inputs
```

1. run prediction on your data

- display the `bioimageio-predict` command interface

  ```console
  > bioimageio predict -h
  usage: bioimageio predict [-h] [--inputs {str,Sequence[str]}] [--outputs {str,Sequence[str]}] [--overwrite bool]
                            [--blockwise bool] [--stats Path]
                            SOURCE

  bioimageio-predict - Run inference on your data with a bioimage.io model.

  positional arguments:
    SOURCE                Url/path to a bioimageio.yaml/rdf.yaml file or a bioimage.io resource identifier, e.g.
                          'affable-shark'

  optional arguments:
    -h, --help            show this help message and exit
    --inputs {str,Sequence[str]}
                          model inputs Either a single path/glob pattern including `{tensor_id}` to be used for all
                          model inputs, or a list of paths/glob patterns for each model input respectively. For models
                          with a single input a single path/glob pattern with `{tensor_id}` is also accepted.
                          (default: model_inputs/*/{tensor_id}.*)
    --outputs {str,Sequence[str]}
                          output paths analog to `inputs` (default: outputs_{model_id}/{sample_id}/{tensor_id}.npy)
    --overwrite bool      allow overwriting existing output files (default: False)
    --blockwise bool      process inputs blockwise (default: False)
    --stats Path          path to dataset statistics (will be written if it does not exist, but the model requires
                          statistical dataset measures) (default: model_inputs\dataset_statistics.json)
  ```

- locate your input data
- predict away!

  ```console
  bioimageio predict affable-shark
  ```

- for convenience the command line arguments may be given in a `bioimageio-cli.json` or `bioimageio-cli.yaml` file.
  The YAML file takes priority over the JSON file.
  Addtional command line arguments take the highest priority.

  ```yaml
  # bioimageio-cli.yaml
  inputs: inputs/*_{tensor_id}.h5
  outputs: outputs_{model_id}/{sample_id}_{tensor_id}.h5
  overwrite: true
  blockwise: true
  stats: inputs/dataset_statistics.json
  ```

  ```console
  bioimageio predict affable-shark
  ```

## Installation

### Via Mamba/Conda

The `bioimageio.core` package can be installed from conda-forge via

```console
mamba install -c conda-forge bioimageio.core
```

If you do not install any additional deep learning libraries, you will only be able to use general convenience
functionality, but not any functionality for model prediction.
To install additional deep learning libraries use:

- Pytorch/Torchscript:

  CPU installation (if you don't have an nvidia graphics card):

  ```console
  mamba install -c pytorch -c conda-forge bioimageio.core pytorch torchvision cpuonly
  ```

  GPU installation (for cuda 11.6, please choose the appropriate cuda version for your system):

  ```console
  mamba install -c pytorch -c nvidia -c conda-forge bioimageio.core pytorch torchvision pytorch-cuda=11.8
  ```

  Note that the pytorch installation instructions may change in the future. For the latest instructions please refer to [pytorch.org](https://pytorch.org/).

- Tensorflow

  Currently only CPU version supported

  ```console
  mamba install -c conda-forge bioimageio.core tensorflow
  ```

- ONNXRuntime

  Currently only cpu version supported

  ```console
  mamba install -c conda-forge bioimageio.core onnxruntime
  ```

### Via pip

The package is also available via pip
(e.g. with recommended extras `onnx` and `pytorch`):

```console
pip install "bioimageio.core[onnx,pytorch]"
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

- [model creation/loading with bioimageio.spec](https://github.com/bioimage-io/spec-bioimage-io/blob/main/example/load_model_and_create_your_own.ipynb)

and the [developer documentation](https://bioimage-io.github.io/core-bioimage-io-python/bioimageio/core.html).

## Model Specification

The model specification and its validation tools can be found at <https://github.com/bioimage-io/spec-bioimage-io>.

## Changelog

### 0.6.9

- improve bioimageio command line interface (details in #157)
  - add `predict` command
  - package command input `path` is now required

### 0.6.8

- testing model inference will now check all weight formats
  (previously only the first one for which model adapter creation succeeded had been checked)
- fix predict with blocking (Thanks @thodkatz)

### 0.6.7

- `predict()` argument `inputs` may be sample

### 0.6.6

- add aliases to match previous API more closely

### 0.6.5

- improve adapter error messages

### 0.6.4

- add `bioimageio validate-format` command
- improve error messages and display of command results

### 0.6.3

- Fix [#386](https://github.com/bioimage-io/core-bioimage-io-python/issues/386)
- (in model inference testing) stop assuming model inputs are tileable

### 0.6.2

- Fix [#384](https://github.com/bioimage-io/core-bioimage-io-python/issues/384)

### 0.6.1

- Fix [#378](https://github.com/bioimage-io/core-bioimage-io-python/pull/378) (with [#379](https://github.com/bioimage-io/core-bioimage-io-python/pull/379))*

### 0.6.0

- add compatibility with new bioimageio.spec 0.5 (0.5.2post1)
- improve interfaces

### 0.5.10

- [Fix critical bug in predict with tiling](https://github.com/bioimage-io/core-bioimage-io-python/pull/359)
