import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import tensorflow
from tensorflow import saved_model

from bioimageio.spec import AnyModel, load_description
from bioimageio.spec._internal.io_utils import download


def _zip_model_bundle(model_bundle_folder: Path):
    zipped_model_bundle = f"{model_bundle_folder}.zip"

    with ZipFile(zipped_model_bundle, "w") as zip_obj:
        for root, _, files in os.walk(model_bundle_folder):
            for filename in files:
                src = os.path.join(root, filename)
                zip_obj.write(src, os.path.relpath(src, model_bundle_folder))

    try:
        shutil.rmtree(model_bundle_folder)
    except Exception:
        print("TensorFlow bundled model was not removed after compression")

    return zipped_model_bundle


# adapted from
# https://github.com/deepimagej/pydeepimagej/blob/master/pydeepimagej/yaml/create_config.py#L236
def _convert_tf1(keras_weight_path: Path, output_path: Path, input_name: str, output_name: str, zip_weights: bool):
    try:
        # try to build the tf model with the keras import from tensorflow
        from tensorflow import keras

    except Exception:
        # if the above fails try to export with the standalone keras
        import keras

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

    build_tf_model()

    if zip_weights:
        output_path = _zip_model_bundle(output_path)
    print("TensorFlow model exported to", output_path)

    return 0


def _convert_tf2(keras_weight_path, output_path, zip_weights):
    try:
        # try to build the tf model with the keras import from tensorflow
        from tensorflow import keras
    except Exception:
        # if the above fails try to export with the standalone keras
        import keras

    model = keras.models.load_model(keras_weight_path)
    keras.models.save_model(model, output_path)

    if zip_weights:
        output_path = _zip_model_bundle(output_path)
    print("TensorFlow model exported to", output_path)

    return 0


def convert_weights_to_tensorflow_saved_model_bundle(
    model_spec: Union[str, Path, AnyModel], output_path: Union[str, Path]
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

    model = load_description(model_spec)
    model.weights.keras_hdf5 is not None
    weight_spec = model.weights.keras_hdf5
    weight_path = download(weight_spec.source).path

    if weight_spec.tensorflow_version:
        model_tf_major_ver = int(weight_spec.tensorflow_version.major)
        if model_tf_major_ver != tf_major_ver:
            raise RuntimeError(f"Tensorflow major versions of model {model_tf_major_ver} is not {tf_major_ver}")

    if tf_major_ver == 1:
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise NotImplementedError(
                "Weight conversion for models with multiple inputs or outputs is not yet implemented."
            )
        return _convert_tf1(weight_path, str(path_), model.inputs[0].name, model.outputs[0].name, zip_weights)
    else:
        return _convert_tf2(weight_path, str(path_), zip_weights)
