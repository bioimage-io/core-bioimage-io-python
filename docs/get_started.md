To get started we recommend installing bioimageio.core with conda together with a deep
learning framework, e.g. pytorch, and run a few `bioimageio` commands to see what
bioimage.core has to offer:

1. Install with conda (for more details on conda environments, [checkout the conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).
Recommended third party deep learning libraries to install alongside bioimageio.core and
[pip installation instructions are here](installation.md).

    ```console
    conda install -c conda-forge bioimageio.core pytorch
    ```

1. Get an overview of available commands

    ```bash exec="1" source="console" result="ansi"
    bioimageio --help
    ```

1. Test a model

    ```bash exec="1" source="console" result="ansi"
    bioimageio test affable-shark
    ```

    To test your model replace the already published model identifier
    'affabl-shark' with a local folder or path to a bioimageio.yaml file.
    Check out the [bioimageio.spec documentation](https://bioimage-io.github.io/spec-bioimage-io) for more information
    on the bioimage.io metadata description format.

    The Python equivalent would be:

    ```python exec="1" souce="console"
    from bioimageio.core import test_description

    summary = test_description("affable-shark")
    summary.display()
    ```

1. Run prediction on your data

- Display the `bioimageio predict` command help to get an overview:

    ```bash exec="1" source="console" result="ansi"
    bioimageio predict --help
    ```

- create an example and run prediction locally!

    ```bash exec="1" source="console" result="ansi"
    bioimageio predict affable-shark --example
    ```

1. For model inference from within Python these options are available:

    - [bioimageio.core.predict][] to run inference on a single sample/image
    - [bioimageio.core.predict_many][] to run inference on a set of samples
    - [bioimageio.core.create_prediction_pipeline][] for reusing the instatiated model and more fine-grain control over the inference process this function creates a suitable [bioimageio.core.PredictionPipeline][] for more advanced use.
