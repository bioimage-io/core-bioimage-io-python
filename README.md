![License](https://img.shields.io/github/license/bioimage-io/core-bioimage-io-python.svg)
[![PyPI](https://img.shields.io/pypi/v/bioimageio-core.svg?style=popout)](https://pypi.org/project/bioimageio.core/)
[![conda-version](https://anaconda.org/conda-forge/bioimageio.core/badges/version.svg)](https://anaconda.org/conda-forge/bioimageio.core/)
[![downloads](https://static.pepy.tech/badge/bioimageio.core)](https://pepy.tech/project/bioimageio.core)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/bioimageio.core.svg?label=conda-forge)](https://anaconda.org/conda-forge/bioimageio.core/)
![code style](https://img.shields.io/badge/code%20style-black-000000.svg)
[![coverage](https://bioimage-io.github.io/core-bioimage-io-python/coverage/coverage-badge.svg)](https://bioimage-io.github.io/core-bioimage-io-python/coverage/index.html)

# bioimageio.core

`bioimageio.core` is a python package that implements prediction with bioimage.io models
including standardized pre- and postprocessing operations.
Such models are represented as [bioimageio.spec](https://bioimage-io.github.io/spec-bioimage-io) resource descriptions.

In addition bioimageio.core provides functionality to convert model weight formats
and compute selected dataset statistics used for preprocessing.

## Documentation

[Here you find the bioimageio.core documentation.](https://bioimage-io.github.io/core-bioimage-io-python)

#### Examples

Notebooks that save and load resource descriptions and validate their format (using <a href="https://bioimage-io.github.io/core-bioimage-io-python/bioimageio/spec.html">bioimageio.spec</a>, a dependency of bioimageio.core)
<ul>
<li><a href="https://github.com/bioimage-io/spec-bioimage-io/blob/main/example/load_model_and_create_your_own.ipynb">load_model_and_create_your_own.ipynb</a> <a target="_blank" href="https://colab.research.google.com/github/bioimage-io/spec-bioimage-io/blob/main/example/load_model_and_create_your_own.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></li>
<li><a href="https://github.com/bioimage-io/spec-bioimage-io/blob/main/example/dataset_creation.ipynb">dataset_creation.ipynb</a> <a target="_blank" href="https://colab.research.google.com/github/bioimage-io/spec-bioimage-io/blob/main/example/dataset_creation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></li>
</ul>

Use the described resources in Python with <a href="https://bioimage-io.github.io/core-bioimage-io-python/bioimageio/core.html">bioimageio.core</a>
<ul>
<li><a href="https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_usage.ipynb">model_usage.ipynb</a><a target="_blank" href="https://colab.research.google.com/github/bioimage-io/core-bioimage-io-python/blob/main/example/model_usage.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></li>
</ul>

#### Presentations

- [Create a model from scratch](https://bioimage-io.github.io/core-bioimage-io-python/presentations/create_ambitious_sloth.slides.html) ([source](https://github.com/bioimage-io/core-bioimage-io-python/tree/main/presentations))

## Set up Development Environment

To set up a development environment run the following commands:

```console
conda create -n core python=$(grep -E '^requires-python' pyproject.toml | grep -oE '[0-9]+\.[0-9]+')
conda activate core
pip install -e .[dev,partners]
```

### Joint development of bioimageio.spec and bioimageio.core

Assuming [spec-bioimage-io](https://github.com/bioimage-io/spec-bioimage-io) is cloned to the parent folder
a joint development environment can be created with the following commands:

```console
conda create -n core python=$(grep -E '^requires-python' pyproject.toml | grep -oE '[0-9]+\.[0-9]+')
conda activate core
pip install -e .[dev,partners] -e ../spec-bioimage-io[dev]
```

## Logging level

`bioimageio.spec` and `bioimageio.core` use [loguru](https://github.com/Delgan/loguru) for logging, hence the logging level
may be controlled with the `LOGURU_LEVEL` environment variable.
The `bioimageio` CLI has logging enabled by default.
To activate logging when using bioimageio.spec/bioimageio.core as a library, add

```python
from loguru import logger

logger.enable("bioimageio")
```

## Changelog

See [changelog.md](changelog.md)
