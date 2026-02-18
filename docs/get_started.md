## Finding a compatible Python environment

For model inference you need a Python environment with the `bioimageio.core` package and model (framework) specific dependencies installed.
You may choose to install `bioimageio.core` alongside (a) suitable framework(s) as optional dependencies with pip, e.g.:

```bash
pip install bioimageio.core[pytorch,onnx]
```

If you are not sure which framework you want to use this model with or the model comes with custom dependencies,
you may choose to have the bioimageio Command Line Interface (CLI) create a suitable environment for a specific model,
using [mini-forge](https://github.com/conda-forge/miniforge) (or your favorite conda distribution).
For more details on conda environments, [checkout the conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
First create/use any conda environment with `bioimageio.core>0.9.6` in it:

```bash
conda create -n bioimageio -c conda-forge "bioimageio.core>0.9.6"
conda activate bioimageio
```

Choose a model source, e.g. a bioimage.io model id like "affable-shark" or a path/url to a bioimageio.yaml (often named rdf.yaml).
Then use the bioimageio CLI (or [bioimageio.core.test_description][]) to test the model.
Use runtime-env=as-described to test each available weight format in the recommended conda environment that is installed on the fly if necessary:

<!-- TODO: execute during doc build -->

```bash
bioimageio test affable-shark --runtime-env=as-described
```

The resulting report shows details of the tests performed in the respective conda environments.
Inspecting the report, choose a conda environment that passed all tests.
The conda environments will be named by the SHA-256 value of the generated conda environment.yaml, e.g. "95227f474ca45b024cf315edb4101e4919199d0a79ef5ff1eb474dc8ce1ec4d8".

You may want to rename or clone your chosen conda environment:

```bash
conda activate base
conda rename -n 95227f474ca45b024cf315edb4101e4919199d0a79ef5ff1eb474dc8ce1ec4d8 bioimageio-affable-shark
conda activate bioimageio-affable-shark
```

## Test model+environment

Test a bioimageio compatible model, e.g. "affable-shark" in an active Python environment:

```bash exec="1" source="console" result="ansi" width="200"
bioimageio test affable-shark
```

To test your model replace the already published model identifier 'affabl-shark' with a local folder or path to a bioimageio.yaml file.
Check out the [bioimageio.spec documentation](https://bioimage-io.github.io/spec-bioimage-io) for more information on the bioimage.io metadata description format.

The Python equivalent would be:

```python exec="1" souce="console" result="ansi" width="300"
from bioimageio.core import test_description

summary = test_description("affable-shark")
summary.display()
```

## CLI: bioimageio predict

You can use the `bioimageio` Command Line Interface (CLI) provided by the `bioimageio.core` package to run prediction with a bioimageio compatible model in a [suitable Python environment](#finding-a-compatible-python-environment).

```bash exec="1" source="console" result="ansi" width="200"
bioimageio predict --help
```

Create a local example and run prediction locally:

```bash exec="1" source="console" result="ansi" width="200"
bioimageio predict affable-shark --example
```

## Python: bioimageio.core.predict

Here is a code snippet to get started deploying a model in Python using the test sample provided by the model description:

```python
from bioimageio.core import load_model_description, predict
from bioimageio.core.digest_spec import get_test_input_sample

model_descr = load_model_description("<model.yaml or model.zip path or URL>")
input_sample = get_test_input_sample(model_descr)
output_sample = predict(model=model_descr, inputs=input_sample)
```

### Python: predict your own data

```python
from bioimageio.core.digest_spec import create_sample_for_model

input_sample = create_sample_for_model(
    model_descr,
    inputs={{"raw": "<path to your input image>"}}
)
output_sample = predict(model=model_descr, inputs=input_sample)
```

### Python: prediction options

For model inference from within Python these options are available:

- [bioimageio.core.predict][] to run inference on a single sample/image.
- [bioimageio.core.predict_many][] to run inference on a set of samples.
- [bioimageio.core.create_prediction_pipeline][] for reusing the instatiated model and more fine-grain control over the inference process this function creates a suitable [bioimageio.core.PredictionPipeline][] for more advanced use.

## Other bioimageio.core functionality

### CLI: bioimageio commands

To get an overview of available commands:

```bash exec="1" source="console" result="ansi" width="200"
bioimageio --help
```

### Python: API docs

See [bioimageio.core][].
