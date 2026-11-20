## bioimageio Command Line Interface

`bioimageio.core` installs a command line interface (CLI) for testing models and other functionality.
You can list all the available commands via:

```bash exec="1" source="console" result="ansi" width="200"
bioimageio --help
```

For concrete examples see [Get started](get-started.md).

### Server devices

The Gradio server selects devices automatically by default. To choose devices, use
`bioimageio server --devices '["cpu"]'` (or specify other supported device names).
In Python, call `bioimageio.core.remote_backends.gradio.server.main(devices=["cpu"])`.
For direct in-process calls to the server handlers, pass `devices=["cpu"]` to
`predict`, `load_model`, or `test_model`. Device availability and selection depend
on the model backend.

### CLI inputs from file

For convenience the command line options (not arguments) may be given in a `bioimageio-cli.json` or `bioimageio-cli.yaml` file, e.g.:

```yaml
# bioimageio-cli.yaml
inputs: inputs/*_{tensor_id}.tiff
outputs: outputs_{model_id}/{sample_id}_{tensor_id}.tiff
overwrite: true
blockwise: true
stats: inputs/dataset_statistics.json
```
