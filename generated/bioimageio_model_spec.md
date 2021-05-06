# BioImage.IO Model Description File Specification 0.3.0
A model entry in the bioimage.io model zoo is defined by a configuration file model.yaml.
The configuration file must contain the following fields; optional fields are indicated by _optional_. 
_optional*_ with an asterisk indicates the field is optional depending on the value in another field.

* `format_version` _String_ Version of the BioImage.IO Model Description File Specification used. 
This is mandatory, and important for the consumer software to verify before parsing the fields. 
The recommended behavior for the implementation is to keep backward compatibility and throw an error if the model yaml 
is in an unsupported format version. The current format version described here is 
0.3.0
* `authors` _List\[String\]_ A list of author strings. 
A string can be separated by `;` in order to identify multiple handles per author.
The authors are the creators of the specifications and the primary points of contact.
* `cite` _List\[CiteEntry\]_ A citation entry or list of citation entries.
Each entry contains a mandatory `text` field and either one or both of `doi` and `url`.
E.g. the citation for the model architecture and/or the training data used. is a Dict with the following keys:
  * `text` _String_ 
  * `doi` _optional String_ 
  * `url` _optional String_ 
* `description` _String_ A string containing a brief description.
* `documentation` _URI→String_ Relative path to file with additional documentation in markdown.
* `license` _String_ A string to a common license name (e.g. `MIT`, `APLv2`) or a relative path to the license file.
* `name` _String_ Name of this model. It should be human-readable and only contain letters, numbers, `_`, `-` or spaces and not be longer than 36 characters.
* `tags` _List\[String\]_ A list of tags.
* `test_inputs` _List\[URI→String\]_ List of URIs to test inputs as described in inputs for a single test case. Supported file formats/extensions: '.npy'
* `test_outputs` _List\[URI→String\]_ Analog to to test_inputs.
* `timestamp` _DateTime_ Timestamp of the initial creation of this model in [ISO 8601](#https://en.wikipedia.org/wiki/ISO_8601) format.
* `weights` _Dict\[String, WeightsEntry\]_ The weights for this model. Weights can be given for different formats, but should otherwise be equivalent. The available weight formats determine which consumers can use this model.
  1. _String_ Format of this set of weights. Weight formats can define additional (optional or required) fields. See [supported_formats_and_operations.md#Weight Format](https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#weight_format). One of: pickle, pytorch_state_dict, pytorch_script, keras_hdf5, tensorflow_js, tensorflow_saved_model_bundle, onnx
  1. _WeightsEntry_  is a Dict with the following keys:
    * `source` _URI→String_ Link to the source file. Preferably a url.
    * `attachments` _optional Dict\[Any, Any\]_ Dictionary of text keys and URI (or a list of URI) values to additional, relevant files that are specific to the current weight format. A list of URIs can be listed under the `files` key to included additional files for generating the model package.
    * `authors` _optional List\[String\]_ A list of authors. If this is the root weight (it does not have a `parent` field): the person(s) that have trained this model. If this is a child weight (it has a `parent` field): the person(s) who have converted the weights to this format.
    * `opset_version` _optional Number_ 
    * `parent` _optional String_ The source weights used as input for converting the weights to this format. For example, if the weights were converted from the format `pytorch_state_dict` to `pytorch_script`, the parent is `pytorch_state_dict`. All weight entries except one (the initial set of weights resulting from training the model), need to have this field.
    * `sha256` _optional String_ SHA256 checksum of the source file specified. You can drag and drop your file to this [online tool](http://emn178.github.io/online-tools/sha256_checksum.html) to generate it in your browser. Or you can generate the SHA256 code for your model and weights by using for example, `hashlib` in Python. 
Code snippet to compute SHA256 checksum

```python
import hashlib

filename = "your filename here"
with open(filename, "rb") as f:
  bytes = f.read() # read entire file as bytes
  readable_hash = hashlib.sha256(bytes).hexdigest()
  print(readable_hash)
  ```


    * `tensorflow_version` _optional StrictVersion→String_ 
* `attachments` _optional* Dict\[String, Union\[URI→String | List\[URI→String\]\]\]_ Dictionary of text keys and URI (or a list of URI) values to additional, relevant 
files. E.g. we can place a list of URIs under the `files` to list images and other files that are necessary for the 
documentation or for the model to run, these files will be included when generating the model package.
* `config` _optional Dict\[Any, Any\]_ 
A custom configuration field that can contain any other keys which are not defined above. It can be very specifc to a framework or specific tool. To avoid conflicted definitions, it is recommended to wrap configuration into a sub-field named with the specific framework or tool name. 

For example:
```yaml
config:
  # custom config for DeepImageJ, see https://github.com/bioimage-io/configuration/issues/23
  deepimagej:
    model_keys:
      # In principle the tag "SERVING" is used in almost every tf model
      model_tag: tf.saved_model.tag_constants.SERVING
      # Signature definition to call the model. Again "SERVING" is the most general
      signature_definition: tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    test_information:  
      input_size: [2048x2048] # Size of the input images  
      output_size: [1264x1264 ]# Size of all the outputs  
      device: cpu # Device used. In principle either cpu or GPU  
      memory_peak: 257.7 Mb # Maximum memory consumed by the model in the device  
      runtime: 78.8s # Time it took to run the model
      pixel_size: [9.658E-4µmx9.658E-4µm] # Size of the pixels of the input
```

* `covers` _optional List\[URI→String\]_ A list of cover images provided by either a relative path to the model folder, or a hyperlink starting with 'https'.Please use an image smaller than 500KB and an aspect ratio width to height of 2:1. The supported image formats are: 'jpg', 'png', 'gif'.
* `dependencies` _optional Dependencies→String_ Dependency manager and dependency file, specified as `<dependency manager>:<relative path to file>`. For example: 'conda:./environment.yaml', 'maven:./pom.xml', or 'pip:./requirements.txt'
* `framework` _optional String_ The deep learning framework of the source code. One of: scikit-learn, pytorch, tensorflow. This field is only required if the field `source` is present.
* `git_repo` _optional String_ A url to the git repository, e.g. to Github or Gitlab.
If the model is contained in a subfolder of a git repository, then a url to the exact folder 
(which contains the configuration yaml file) should be used.
* `inputs` _List\[InputTensor\]_ Describes the input tensors expected by this model. is a Dict with the following keys:
  * `axes` _Axes→String_ Axes identifying characters from: bitczyx. Same length and order as the axes in `shape`.
        
    | character | description |
    | --- | --- |
    |  b  |  batch (groups multiple samples) |
    |  i  |  instance/index/element |
    |  t  |  time |
    |  c  |  channel |
    |  z  |  spatial dimension z |
    |  y  |  spatial dimension y |
    |  x  |  spatial dimension x |
  * `data_type` _String_ The data type of this tensor. For inputs, only `float32` is allowed and the consumer software needs to ensure that the correct data type is passed here. For outputs can be any of `float32, float64, (u)int8, (u)int16, (u)int32, (u)int64`. The data flow in bioimage.io models is explained [in this diagram.](https://docs.google.com/drawings/d/1FTw8-Rn6a6nXdkZ_SkMumtcjvur9mtIhRqLwnKqZNHM/edit).
  * `name` _String_ Tensor name.
  * `shape` _InputShape→Union\[ExplicitShape→List\[Integer\] | ImplicitInputShape\]_ Specification of tensor shape.
    1. _optional ExplicitShape→List\[Integer\]_ Exact shape with same length as `axes`, e.g. `shape: [1, 512, 512, 1]`
    1. _ImplicitInputShape_ A sequence of valid shapes given by `shape = min + k * step for k in {0, 1, ...}`. is a Dict with the following keys:
      * `min` _List\[Integer\]_ The minimum input shape with same length as `axes`
      * `step` _List\[Integer\]_ The minimum shape change with same length as `axes`
  * `data_range` _optional Tuple_ Tuple `(minimum, maximum)` specifying the allowed range of the data in this tensor. If not specified, the full data range that can be expressed in `data_type` is allowed.
  * `description` _optional String_ 
  * `preprocessing` _optional List\[Preprocessing\]_ Description of how this input should be preprocessed.
* `kwargs` _optional Kwargs→Dict\[String, Any\]_ Keyword arguments for the implementation specified by `source`. This field is only required if the field `source` is present.
* `language` _optional* String_ Programming language of the source code. One of: python, java. This field is only required if the field `source` is present.
* `outputs` _List\[OutputTensor\]_ Describes the output tensors from this model. is a Dict with the following keys:
  * `axes` _Axes→String_ Axes identifying characters from: bitczyx. Same length and order as the axes in `shape`.
        
    | character | description |
    | --- | --- |
    |  b  |  batch (groups multiple samples) |
    |  i  |  instance/index/element |
    |  t  |  time |
    |  c  |  channel |
    |  z  |  spatial dimension z |
    |  y  |  spatial dimension y |
    |  x  |  spatial dimension x |
  * `data_type` _String_ The data type of this tensor. For inputs, only `float32` is allowed and the consumer software needs to ensure that the correct data type is passed here. For outputs can be any of `float32, float64, (u)int8, (u)int16, (u)int32, (u)int64`. The data flow in bioimage.io models is explained [in this diagram.](https://docs.google.com/drawings/d/1FTw8-Rn6a6nXdkZ_SkMumtcjvur9mtIhRqLwnKqZNHM/edit).
  * `name` _String_ Tensor name.
  * `shape` _OutputShape→Union\[ExplicitShape→List\[Integer\] | ImplicitOutputShape\]_ 
    1. _optional ExplicitShape→List\[Integer\]_ 
    1. _ImplicitOutputShape_ In reference to the shape of an input tensor, the shape of the output tensor is `shape = shape(input_tensor) * scale + 2 * offset`. is a Dict with the following keys:
      * `offset` _List\[Integer\]_ Position of origin wrt to input.
      * `reference_input` _String_ Name of the reference input tensor.
      * `scale` _List\[Float\]_ 'output_pix/input_pix' for each dimension.
  * `data_range` _optional Tuple_ Tuple `(minimum, maximum)` specifying the allowed range of the data in this tensor. If not specified, the full data range that can be expressed in `data_type` is allowed.
  * `description` _optional String_ 
  * `halo` _optional List\[Integer\]_ The halo to crop from the output tensor (for example to crop away boundary effects or for tiling). The halo should be cropped from both sides, i.e. `shape_after_crop = shape - 2 * halo`. The `halo` is not cropped by the bioimage.io model, but is left to be cropped by the consumer software. Use `shape:offset` if the model output itself is cropped and input and output shapes not fixed.
  * `postprocessing` _optional List\[Postprocessing\]_ Description of how this output should be postprocessed.
* `packaged_by` _optional List\[String\]_ The persons that have packaged and uploaded this model. Only needs to be specified if different from `authors` in root or any WeightsEntry.
* `parent` _ModelParent_ Parent model from which the trained weights of this model have been derived, e.g. by finetuning the weights of this model on a different dataset. For format changes of the same trained model checkpoint, see `weights`. is a Dict with the following keys:
  * `sha256` _optional SHA256→String_ Hash of the weights of the parent model.
  * `uri` _optional URI→String_ Url of another model available on bioimage.io or path to a local model in the bioimage.io specification. If it is a url, it needs to be a github url linking to the page containing the model (NOT the raw file).
* `run_mode` _RunMode_ Custom run mode for this model: for more complex prediction procedures like test time data augmentation that currently cannot be expressed in the specification. The different run modes should be listed in [supported_formats_and_operations.md#Run Modes](https://github.com/bioimage-io/configuration/blob/master/supported_formats_and_operations.md#run-modes). is a Dict with the following keys:
  * `name` _String_ The name of the `run_mode`
  * `kwargs` _optional Kwargs→Dict\[String, Any\]_ Key word arguments.
* `sample_inputs` _optional List\[URI→String\]_ List of URIs to sample inputs to illustrate possible inputs for the model, for example stored as png or tif images.
* `sample_outputs` _optional List\[URI→String\]_ List of URIs to sample outputs corresponding to the `sample_inputs`.
* `sha256` _optional String_ SHA256 checksum of the model source code file.You can drag and drop your file to this [online tool](http://emn178.github.io/online-tools/sha256_checksum.html) to generate it in your browser. Or you can generate the SHA256 code for your model and weights by using for example, `hashlib` in Python. 
Code snippet to compute SHA256 checksum

```python
import hashlib

filename = "your filename here"
with open(filename, "rb") as f:
  bytes = f.read() # read entire file as bytes
  readable_hash = hashlib.sha256(bytes).hexdigest()
  print(readable_hash)
  ```

 This field is only required if the field source is present.
* `source` _optional* ImportableSource→String_ Language and framework specific implementation. As some weights contain the model architecture, the source is optional depending on the present weight formats. `source` can either point to a local implementation: `<relative path to file>:<identifier of implementation within the source file>` or the implementation in an available dependency: `<root-dependency>.<sub-dependency>.<identifier>`.
For example: `./my_function:MyImplementation` or `core_library.some_module.some_function`.
