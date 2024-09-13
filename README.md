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
bioimage.core has to offer:

1. install with conda (for more details on conda environments, [checkout the conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

    ```console
    conda install -c conda-forge bioimageio.core pytorch
    ```

1. test a model

    ```console
    $ bioimageio test powerful-chipmunk
    ...
    ```

    <details>
    <summary>(Click to expand output)</summary>

    ```console


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

    </details>

    or

    ```console
    $ bioimageio test impartial-shrimp
    ...
    ```

    <details><summary>(Click to expand output)</summary>

    ```console
      ✔️                 bioimageio validation passed
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      source            https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/impartial-shrimp/1.1/files/rdf.yaml
      format version    model 0.5.3
      bioimageio.spec   0.5.3.2
      bioimageio.core   0.6.9


      ❓   location                    detail
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ✔️                                initialized ModelDescr to describe model 0.5.3


      ✔️                                bioimageio.spec format validation model 0.5.3

      🔍   context.perform_io_checks   False
      🔍   context.warning_level       error

      ✔️                                Reproduce test outputs from test inputs (pytorch_state_dict)


      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 1 and size parameter n:

                                      0

      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 2 and size parameter n:

                                      0

      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 1 and size parameter n:

                                      1

      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 2 and size parameter n:

                                      1

      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 1 and size parameter n:

                                      2

      ✔️                                Run pytorch_state_dict inference for inputs with batch_size: 2 and size parameter n:

                                      2

      ✔️                                Reproduce test outputs from test inputs (torchscript)


      ✔️                                Run torchscript inference for inputs with batch_size: 1 and size parameter n: 0


      ✔️                                Run torchscript inference for inputs with batch_size: 2 and size parameter n: 0


      ✔️                                Run torchscript inference for inputs with batch_size: 1 and size parameter n: 1


      ✔️                                Run torchscript inference for inputs with batch_size: 2 and size parameter n: 1


      ✔️                                Run torchscript inference for inputs with batch_size: 1 and size parameter n: 2


      ✔️                                Run torchscript inference for inputs with batch_size: 2 and size parameter n: 2
    ```

    </details>
1. run prediction on your data

- display the `bioimageio-predict` command help to get an overview:

    ```console
    $ bioimageio predict --help
    ...
    ```

    <details>
    <summary>(Click to expand output)</summary>

    ```console
    usage: bioimageio predict [-h] [--inputs Sequence[Union[str,Annotated[Tuple[str,...],MinLenmin_length=1]]]]
                              [--outputs {str,Tuple[str,...]}] [--overwrite bool] [--blockwise bool] [--stats Path]
                              [--preview bool]
                              [--weight_format {typing.Literal['keras_hdf5','onnx','pytorch_state_dict','tensorflow_js','tensorflow_saved_model_bundle','torchscript'],any}]
                              [--example bool]
                              SOURCE

    bioimageio-predict - Run inference on your data with a bioimage.io model.

    positional arguments:
      SOURCE                Url/path to a bioimageio.yaml/rdf.yaml file
                            or a bioimage.io resource identifier, e.g. 'affable-shark'

    optional arguments:
      -h, --help            show this help message and exit
      --inputs Sequence[Union[str,Annotated[Tuple[str,...],MinLen(min_length=1)]]]
                            Model input sample paths (for each input tensor)

                            The input paths are expected to have shape...
                            - (n_samples,) or (n_samples,1) for models expecting a single input tensor
                            - (n_samples,) containing the substring '{input_id}', or
                            - (n_samples, n_model_inputs) to provide each input tensor path explicitly.

                            All substrings that are replaced by metadata from the model description:
                            - '{model_id}'
                            - '{input_id}'

                            Example inputs to process sample 'a' and 'b'
                            for a model expecting a 'raw' and a 'mask' input tensor:
                            --inputs="[["a_raw.tif","a_mask.tif"],["b_raw.tif","b_mask.tif"]]"
                            (Note that JSON double quotes need to be escaped.)

                            Alternatively a `bioimageio-cli.yaml` (or `bioimageio-cli.json`) file
                            may provide the arguments, e.g.:
                            ```yaml
                            inputs:
                            - [a_raw.tif, a_mask.tif]
                            - [b_raw.tif, b_mask.tif]
                            ```

                            `.npy` and any file extension supported by imageio are supported.
                            Aavailable formats are listed at
                            https://imageio.readthedocs.io/en/stable/formats/index.html#all-formats.
                            Some formats have additional dependencies.

                              (default: ('{input_id}/001.tif',))
      --outputs {str,Tuple[str,...]}
                            Model output path pattern (per output tensor)

                            All substrings that are replaced:
                            - '{model_id}' (from model description)
                            - '{output_id}' (from model description)
                            - '{sample_id}' (extracted from input paths)

                              (default: outputs_{model_id}/{output_id}/{sample_id}.tif)
      --overwrite bool      allow overwriting existing output files (default: False)
      --blockwise bool      process inputs blockwise (default: False)
      --stats Path          path to dataset statistics
                            (will be written if it does not exist,
                            but the model requires statistical dataset measures)
                              (default: dataset_statistics.json)
      --preview bool        preview which files would be processed
                            and what outputs would be generated. (default: False)
      --weight_format {typing.Literal['keras_hdf5','onnx','pytorch_state_dict','tensorflow_js','tensorflow_saved_model_bundle','torchscript'],any}
                            The weight format to use. (default: any)
      --example bool        generate and run an example

                            1. downloads example model inputs
                            2. creates a `{model_id}_example` folder
                            3. writes input arguments to `{model_id}_example/bioimageio-cli.yaml`
                            4. executes a preview dry-run
                            5. executes prediction with example input

                              (default: False)
    ```

    </details>

- create an example and run prediction locally!

    ```console
    $ bioimageio predict impartial-shrimp --example=True
    ...
    ```

    <details>
    <summary>(Click to expand output)</summary>

    ```console
    🛈 bioimageio prediction preview structure:
    {'{sample_id}': {'inputs': {'{input_id}': '<input path>'},
                    'outputs': {'{output_id}': '<output path>'}}}
    🔎 bioimageio prediction preview output:
    {'1': {'inputs': {'input0': 'impartial-shrimp_example/input0/001.tif'},
          'outputs': {'output0': 'impartial-shrimp_example/outputs/output0/1.tif'}}}
    predict with impartial-shrimp: 100%|███████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.76s/sample]
    🎉 Sucessfully ran example prediction!
    To predict the example input using the CLI example config file impartial-shrimp_example\bioimageio-cli.yaml, execute `bioimageio predict` from impartial-shrimp_example:
    $ cd impartial-shrimp_example
    $ bioimageio predict "impartial-shrimp"

    Alternatively run the following command in the current workind directory, not the example folder:
    $ bioimageio predict --preview=False --overwrite=True --stats="impartial-shrimp_example/dataset_statistics.json" --inputs="[[\"impartial-shrimp_example/input0/001.tif\"]]" --outputs="impartial-shrimp_example/outputs/{output_id}/{sample_id}.tif" "impartial-shrimp"
    (note that a local 'bioimageio-cli.json' or 'bioimageio-cli.yaml' may interfere with this)
  ```

  </details>

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

### CLI inputs from file

For convenience the command line options (not arguments) may be given in a `bioimageio-cli.json`
or `bioimageio-cli.yaml` file, e.g.:

```yaml
# bioimageio-cli.yaml
inputs: inputs/*_{tensor_id}.h5
outputs: outputs_{model_id}/{sample_id}_{tensor_id}.h5
overwrite: true
blockwise: true
stats: inputs/dataset_statistics.json
```

## 🐍 Use in Python

`bioimageio.core` is a python package that implements prediction with bioimageio models
including standardized pre- and postprocessing operations.
These models are described by---and can be loaded with---the bioimageio.spec package.

In addition bioimageio.core provides functionality to convert model weight formats.

To get an overview of this functionality, check out these example notebooks:

- [model creation/loading with bioimageio.spec](https://github.com/bioimage-io/spec-bioimage-io/blob/main/example/load_model_and_create_your_own.ipynb)

and the [developer documentation](https://bioimage-io.github.io/core-bioimage-io-python/bioimageio/core.html).

## Logging level

`bioimageio.spec` and `bioimageio.core` use [loguru](https://github.com/Delgan/loguru) for logging, hence the logging level
may be controlled with the `LOGURU_LEVEL` environment variable.

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
