# BioImage.IO Model Description File Specification 0.3.0
A model entry in the bioimage.io model zoo is defined by a configuration file model.yaml.
The configuration file must contain the following fields; optional fields are followed by [optional]. 
If a field is followed by [optional]*, they are optional depending on another field.

* `format_version` Version of this BioImage.IO Model Description File Specification. This is mandatory, and important for the consumer software to verify before parsing the fields. The recommended behavior for the implementation is to keep backward compatibility, and throw error if the model yaml is in an unsupported format version. Current format version: 0.3.0
* `authors` List
* `cite` CiteEntry
  * `text` String
  * `doi` [optional] String
  * `url` [optional] String
* `description` String
* `documentation` URI
* `framework` String
* `license` String
* `name` String
* `tags` List
* `test_inputs` List
* `test_outputs` List
* `timestamp` DateTime
* `weights` Dict
* `attachments` [optional] Dict
* `config` [optional] Dict
* `covers` [optional] List
* `dependencies` [optional] Dependencies
* `git_repo` [optional] String
* `inputs` [optional] InputTensor
  * `axes` Axes
  * `data_type` String
  * `name` String
  * `shape` InputShape
    1. [optional] ExplicitShape
    1. [optional] ImplicitInputShape
      * `min` List
      * `step` List
  * `data_range` [optional] Tuple
  * `description` [optional] String
  * `preprocessing` [optional] List
* `kwargs` [optional] Dict
* `language` [optional] * Programming language of the source code. For now, we support python and java. This field is only required if the field `source` is present.
* `outputs` [optional] OutputTensor
  * `axes` Axes
  * `data_type` String
  * `name` String
  * `shape` OutputShape
    1. [optional] ExplicitShape
    1. [optional] ImplicitOutputShape
      * `offset` List
      * `reference_input` String
      * `scale` List
  * `data_range` [optional] Tuple
  * `description` [optional] String
  * `halo` [optional] Halo
  * `postprocessing` [optional] List
* `run_mode` [optional] Custom run mode for this model: for more complex prediction procedures like test time data augmentation that currently cannot be expressed in the specification. The different run modes should be listed in [supported_formats_and_operations.md#Run Modes](https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#run-modes).
  * `name` String
  * `kwargs` [optional] Dict
* `sample_inputs` [optional] List
* `sample_outputs` [optional] List
* `sha256` [optional] String
* `source` [optional] * Language and framework specific implementation. As some weights contain the model architecture, the source is optional depending on the present weight formats. `source` can either point to a local implementation: `<relative path to file>:<identifier of implementation within the source file>` or the implementation in an available dependency: `<root-dependency>.<sub-dependency>.<identifier>`.
For example: `./my_function:MyImplementation` or `core_library.some_module.some_function`
