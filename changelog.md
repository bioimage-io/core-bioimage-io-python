### 0.9.6

- bump bioimageio.spec library version to 0.5.7.4
- increase default reprducibility tolerance
- unify quantile (vs percentile) variable names
- add quantile computation method parameter
- accept `SampleQuantile` or `DatasetQuantile` as `min`/`max` arguments to `proc_ops.Clip`
- save actual output during model testing only if an explicit working directory was specified to produce less clutter

### 0.9.5

- bump bioimageio.spec library version to 0.5.6.0
- improved ONNX export from pytorch state dict or torchscript using torch dynamo
- default `opset_version` for `pytorch_to_onnx`/`torchscript_for_onnx` conversions bumped to 18.

### 0.9.4

- bump bioimageio.spec library version to 0.5.5.6
- Replace `conda activate <env name>` with `conda run -n <env name> python --version` when checking if a conda environment exists
  (This is closer to the actual `conda run` command we need and avoids requests by conda to rerun `conda init` (in CI).)
- remove upper xarray pin (use ops from xarray.computation.ops, fallback to xarray.core.ops for older xarray versions)

### 0.9.3

- bump bioimageio.spec library version to 0.5.5.5
- more robust test model reporting
- improved user input axis intepretation
- fixed conda subprocess calls

### 0.9.2

- fix model inference tolerance reporting

### 0.9.1

- fixes:
  - CLI
    - improved handling of summary argument to not create a path with brackets when given a list of paths.
    - improved backward compatibility when runnig tests for models specifying an older bioimageio.core version in their environment.
    This is relevant when using `runtime_env="as-described"`.
    It works by simply trying option `--summary` (new option name) and `--summary-path` (outdated option name)

### 0.9.0

- update to [bioimageio.spec 0.5.4.3](https://github.com/bioimage-io/spec-bioimage-io/blob/main/changelog.md#bioimageiospec-0543)

### 0.8.0

- breaking: removed `decimals` argument from bioimageio CLI and `bioimageio.core.commands.test()`
- New feature: `bioimageio.core.test_description` accepts **runtime_env** and **run_command** to test a resource
  using the conda environment described by that resource (or another specified conda env)
- new CLI command: `bioimageio add-weights` (and utility function: bioimageio.core.add_weights)
- removed `bioimageio.core.proc_ops.get_proc_class` in favor of `bioimageio.core.proc_ops.get_proc`
- new CLI command: `bioimageio update-format`
- new CLI command: `bioimageio update-hashes`

### 0.7.0

- breaking:
  - bioimageio CLI now has implicit boolean flags
- non-breaking:
  - use new `ValidationDetail.recommended_env` in `ValidationSummary`
  - improve `get_io_sample_block_metas()`
    - now works for sufficiently large, but not exactly shaped inputs
  - update to support `zipfile.ZipFile` object with bioimageio.spec==0.5.3.5
  - add io helpers `resolve` and `resolve_and_extract`
  - added `enable_determinism` function and **determinism** input argument for testing with seeded
    random generators and optionally (determinsim=="full") instructing DL frameworks to use
    deterministic algorithms.

### 0.6.10

- fix #423

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
