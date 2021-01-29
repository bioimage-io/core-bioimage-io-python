# python-bioimage-io
Python specific core utilities for interpretation of specification files of the model zoo


## Installation

<!--
TODO from pip/conda
-->


Install from source and in development mode
```
pip install git+https://github.com/bioimage-io/python-bioimage-io
```

## Usage

You can verify a model configuration in the [bioimage.io model format]() using the following command:
```
python -m pybio.spec verify-spec <MY-MODEL>.model.yaml
```
The output of this command will indicate missing or invalid fields in the model file. For example, if the field `timestamp` was missing it would print the following:
```
{'timestamp': ['Missing data for required field.']}
```
or if the field `test_inputs` does not contain a list, it would print:
```
{'test_inputs': ['Not a valid list.']}.
```
If there is no ouput the model configuration is valid.
