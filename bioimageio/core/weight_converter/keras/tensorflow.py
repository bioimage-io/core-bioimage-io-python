import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import bioimageio.spec as spec
from bioimageio.core import load_resource_description

import tensorflow
from tensorflow import saved_model


# adapted from
# https://github.com/deepimagej/pydeepimagej/blob/master/pydeepimagej/yaml/create_config.py#L236
def _convert_tf1(keras_weight_path, output_path, input_name, output_name, zip_weights):
    def build_tf_model():
        keras_model = keras.models.load_model(keras_weight_path)

        builder = saved_model.builder.SavedModelBuilder(output_path)
        signature = saved_model.signature_def_utils.predict_signature_def(
            inputs={input_name: keras_model.input}, outputs={output_name: keras_model.output}
        )

        signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

        builder.add_meta_graph_and_variables(
            keras.backend.get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map
        )
        builder.save()

    try:
        # try to build the tf model with the keras import from tensorflow
        from tensorflow import keras
        build_tf_model()
    except Exception:
        # if the above fails try to export with the standalone keras
        import keras

        build_tf_model()

    if zip_weights:
        zipped_model = f"{output_path}.zip"
        # zip the weights
        file_paths = []
        for folder_names, subfolder, filenames in os.walk(os.path.join(output_path)):
            for filename in filenames:
                # create complete filepath of file in directory
                file_paths.append(os.path.join(folder_names, filename))

        with ZipFile(zipped_model, "w") as zip_obj:
            for f in file_paths:
                # Add file to zip
                zip_obj.write(f, os.path.relpath(f, output_path))

        try:
            shutil.rmtree(output_path)
        except Exception:
            print("TensorFlow bundled model was not removed after compression")
        print("TensorFlow model exported to", zipped_model)
    else:
        print("TensorFlow model exported to", output_path)
    return 0


def convert_weights_to_tensorflow_saved_model_bundle(
    model_spec: Union[str, Path, spec.model.raw_nodes.Model], output_path: Union[str, Path]
):
    """Convert model weights from format 'keras_hdf5' to 'tensorflow_saved_model_bundle'.

    Adapted from
    https://github.com/deepimagej/pydeepimagej/blob/5aaf0e71f9b04df591d5ca596f0af633a7e024f5/pydeepimagej/yaml/create_config.py

    Args:
        model_spec: location of the resource for the input bioimageio model
        output_path: where to save the tensorflow weights. This path must not exist yet.
    """
    tf_major_ver = int(tensorflow.__version__.split(".")[0])

    path_ = Path(output_path)
    if path_.suffix == ".zip":
        path_ = Path(os.path.splitext(path_)[0])
        zip_weights = True
    else:
        zip_weights = False

    if path_.exists():
        raise ValueError(f"The ouptut directory at {path_} must not exist.")

    model = load_resource_description(model_spec)
    assert "keras_hdf5" in model.weights
    weight_spec = model.weights["keras_hdf5"]
    weight_path = str(weight_spec.source)

    if weight_spec.tensorflow_version:
        model_tf_major_ver = weight_spec.tensorflow_version.version[0]
        if model_tf_major_ver != tf_major_ver:
            raise RuntimeError(f"Tensorflow major versions of model {model_tf_major_ver} is not {tf_major_ver}")

    if tf_major_ver == 1:
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise NotImplementedError(
                "Weight conversion for models with multiple inputs or outputs is not yet implemented."
            )
        return _convert_tf1(weight_path, str(path_), model.inputs[0].name, model.outputs[0].name, zip_weights)
    else:
        raise NotImplementedError("Weight conversion for tensorflow 2 is not yet implemented.")
