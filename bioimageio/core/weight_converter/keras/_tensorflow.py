# type: ignore  # TODO: type
import os
import shutil
from pathlib import Path
from typing import no_type_check
from zipfile import ZipFile

try:
    import tensorflow.saved_model
except Exception:
    tensorflow = None

from bioimageio.spec._internal.io_utils import download
from bioimageio.spec.model.v0_5 import ModelDescr


def _zip_model_bundle(model_bundle_folder: Path):
    zipped_model_bundle = model_bundle_folder.with_suffix(".zip")

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
def _convert_tf1(
    keras_weight_path: Path,
    output_path: Path,
    input_name: str,
    output_name: str,
    zip_weights: bool,
):
    try:
        # try to build the tf model with the keras import from tensorflow
        from bioimageio.core.weight_converter.keras._tensorflow import (
            keras,  # type: ignore
        )

    except Exception:
        # if the above fails try to export with the standalone keras
        import keras

    @no_type_check
    def build_tf_model():
        keras_model = keras.models.load_model(keras_weight_path)
        assert tensorflow is not None
        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_path)
        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={input_name: keras_model.input},
            outputs={output_name: keras_model.output},
        )

        signature_def_map = {
            tensorflow.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }

        builder.add_meta_graph_and_variables(
            keras.backend.get_session(),
            [tensorflow.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
        )
        builder.save()

    build_tf_model()

    if zip_weights:
        output_path = _zip_model_bundle(output_path)
    print("TensorFlow model exported to", output_path)

    return 0


def _convert_tf2(keras_weight_path: Path, output_path: Path, zip_weights: bool):
    try:
        # try to build the tf model with the keras import from tensorflow
        from bioimageio.core.weight_converter.keras._tensorflow import keras
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
    model: ModelDescr, output_path: Path
):
    """Convert model weights from format 'keras_hdf5' to 'tensorflow_saved_model_bundle'.

    Adapted from
    https://github.com/deepimagej/pydeepimagej/blob/5aaf0e71f9b04df591d5ca596f0af633a7e024f5/pydeepimagej/yaml/create_config.py

    Args:
        model: The bioimageio model description
        output_path: where to save the tensorflow weights. This path must not exist yet.
    """
    assert tensorflow is not None
    tf_major_ver = int(tensorflow.__version__.split(".")[0])

    if output_path.suffix == ".zip":
        output_path = output_path.with_suffix("")
        zip_weights = True
    else:
        zip_weights = False

    if output_path.exists():
        raise ValueError(f"The ouptut directory at {output_path} must not exist.")

    if model.weights.keras_hdf5 is None:
        raise ValueError("Missing Keras Hdf5 weights to convert from.")

    weight_spec = model.weights.keras_hdf5
    weight_path = download(weight_spec.source).path

    if weight_spec.tensorflow_version:
        model_tf_major_ver = int(weight_spec.tensorflow_version.major)
        if model_tf_major_ver != tf_major_ver:
            raise RuntimeError(
                f"Tensorflow major versions of model {model_tf_major_ver} is not {tf_major_ver}"
            )

    if tf_major_ver == 1:
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise NotImplementedError(
                "Weight conversion for models with multiple inputs or outputs is not yet implemented."
            )
        return _convert_tf1(
            weight_path,
            output_path,
            model.inputs[0].id,
            model.outputs[0].id,
            zip_weights,
        )
    else:
        return _convert_tf2(weight_path, output_path, zip_weights)
