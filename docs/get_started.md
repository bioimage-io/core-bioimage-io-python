To get started we recommend installing bioimageio.core with conda together with a deep
learning framework, e.g. pytorch, and run a few `bioimageio` commands to see what
bioimage.core has to offer:

1. Install with conda (for more details on conda environments, [checkout the conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).
More details and pip instructions are [here](installation.md).

    ```console
    conda install -c conda-forge bioimageio.core pytorch
    ```

1. Get an overview of available commands

    ```console exec="1"
    bioimageio --help
    ```

1. Test a model

    ```console exec="1"
    bioimageio test affable-shark
    ```

    To test your model replace the already published 'affabl-shark' with a local folder or path to a bioimageio.yaml file.
    Check out the [bioimageio.spec documentation](https://bioimage-io.github.io/spec-bioimage-io) for more information
    on the bioimage.io metadata description format.

1. Run prediction on your data

- Display the `bioimageio predict` command help to get an overview:

    ```console exec="1"
    bioimageio predict --help
    ```

- create an example and run prediction locally!

    ```console exec="1"
    bioimageio predict affable-shark --example=True
    ```
