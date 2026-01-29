## bioimageio Command Line Interface

`bioimageio.core` installs a command line interface (CLI) for testing models and other functionality.
You can list all the available commands via:

```console exec="1"
bioimageio --help
```

For concrete examples see [Get started](get-started.md).

### CLI inputs from file

For convenience the command line options (not arguments) may be given in a `bioimageio-cli.json` or `bioimageio-cli.yaml` file, e.g.:

```yaml
# bioimageio-cli.yaml
inputs: inputs/*_{tensor_id}.h5
outputs: outputs_{model_id}/{sample_id}_{tensor_id}.h5
overwrite: true
blockwise: true
stats: inputs/dataset_statistics.json
```
