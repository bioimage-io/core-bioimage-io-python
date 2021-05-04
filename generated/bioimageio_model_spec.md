# BioImage.IO Model Description File Specification 0.3.0
A model entry in the bioimage.io model zoo is defined by a configuration file model.yaml.
The configuration file must contain the following fields; optional fields are indicated by _optional_. 
_optional*_ with an asterisk indicates the field is optional depending on the value in another field.

* `format_version` _String_ Version of the BioImage.IO Model Description File Specification used. This is mandatory, and important for the consumer software to verify before parsing the fields. The recommended behavior for the implementation is to keep backward compatibility and throw an error if the model yaml is in an unsupported format version. The current format version described here is 0.3.0
* `authors` _List\[String\]_ 
* `cite` _List\[CiteEntry\]_  
Each CiteEntry is a Dict with the following keys:
  * `text` _String_ 
  * `doi` _optional String_ 
  * `url` _optional String_ 
* `description` _String_ 
* `documentation` _URI->String_ 
* `framework` _String_ 
* `license` _String_ 
* `name` _String_ 
* `tags` _List\[String\]_ 
* `test_inputs` _List\[URI->String\]_ 
* `test_outputs` _List\[URI->String\]_ 
* `timestamp` _DateTime_ 
* `weights` _Dict\[String, WeightsEntry\]_ 
  1. _String_ weights format. One of: pickle, pytorch_state_dict, pytorch_script, keras_hdf5, tensorflow_js, tensorflow_saved_model_bundle, onnx
  1. _WeightsEntry_  
Each WeightsEntry is a Dict with the following keys:
    * `source` _URI->String_ 
    * `attachments` _optional Dict\[Any, Any\]_ 
    * `authors` _optional List\[String\]_ 
    * `opset_version` _optional Number_ 
    * `parent` _optional String_ 
    * `sha256` _optional String_ 
    * `tensorflow_version` _optional StrictVersion->String_ 
* `attachments` _optional Dict\[String, URI->String\]_ 
* `config` _optional Dict\[Any, Any\]_ 
* `covers` _optional List\[URI->String\]_ 
* `dependencies` _optional Dependencies->String_ Dependency manager and dependency file, specified as `<dependency manager>:<relative path to file>`. For example: 'conda:./environment.yaml', 'maven:./pom.xml', or 'pip:./requirements.txt'
* `git_repo` _optional String_ 
* `inputs` _List\[InputTensor\]_  
Each InputTensor is a Dict with the following keys:
  * `axes` _Axes->String_ 
  * `data_type` _String_ 
  * `name` _String_ 
  * `shape` _InputShape->Union\[ExplicitShape->List\[Integer\] | ImplicitInputShape\]_ 
    1. _optional ExplicitShape->List\[Integer\]_ 
    1. _ImplicitInputShape_  
Each ImplicitInputShape is a Dict with the following keys:
      * `min` _List\[Integer\]_ 
      * `step` _List\[Integer\]_ 
  * `data_range` _optional Tuple_ 
  * `description` _optional String_ 
  * `preprocessing` _optional List\[Preprocessing\]_ 
* `kwargs` _optional Dict\[String, Any\]_ 
* `language` _optional* String_ Programming language of the source code. For now, we support python and java. This field is only required if the field `source` is present.
* `outputs` _List\[OutputTensor\]_  
Each OutputTensor is a Dict with the following keys:
  * `axes` _Axes->String_ 
  * `data_type` _String_ 
  * `name` _String_ 
  * `shape` _OutputShape->Union\[ExplicitShape->List\[Integer\] | ImplicitOutputShape\]_ 
    1. _optional ExplicitShape->List\[Integer\]_ 
    1. _ImplicitOutputShape_  
Each ImplicitOutputShape is a Dict with the following keys:
      * `offset` _List\[Integer\]_ 
      * `reference_input` _String_ 
      * `scale` _List\[Float\]_ 
  * `data_range` _optional Tuple_ 
  * `description` _optional String_ 
  * `halo` _optional Halo->List\[Integer\]_ 
  * `postprocessing` _optional List\[Postprocessing\]_ 
* `run_mode` _RunMode_ Custom run mode for this model: for more complex prediction procedures like test time data augmentation that currently cannot be expressed in the specification. The different run modes should be listed in [supported_formats_and_operations.md#Run Modes](https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#run-modes). 
Each RunMode is a Dict with the following keys:
  * `name` _String_ 
  * `kwargs` _optional Dict\[String, Any\]_ 
* `sample_inputs` _optional List\[URI->String\]_ 
* `sample_outputs` _optional List\[URI->String\]_ 
* `sha256` _optional String_ 
* `source` _optional* ImportableSource->String_ Language and framework specific implementation. As some weights contain the model architecture, the source is optional depending on the present weight formats. `source` can either point to a local implementation: `<relative path to file>:<identifier of implementation within the source file>` or the implementation in an available dependency: `<root-dependency>.<sub-dependency>.<identifier>`.
For example: `./my_function:MyImplementation` or `core_library.some_module.some_function`
