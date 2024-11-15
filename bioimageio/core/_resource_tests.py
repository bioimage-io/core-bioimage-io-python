import traceback
import warnings
from itertools import product
from typing import Dict, Hashable, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
from loguru import logger

from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    build_description,
    dump_description,
    load_description,
)
from bioimageio.spec._internal.common_nodes import ResourceDescrBase
from bioimageio.spec.common import BioimageioYamlContent, PermissiveFileSource
from bioimageio.spec.get_conda_env import get_conda_env
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import WeightsFormat
from bioimageio.spec.summary import (
    ErrorEntry,
    InstalledPackage,
    ValidationDetail,
    ValidationSummary,
)

from ._prediction_pipeline import create_prediction_pipeline
from .axis import AxisId, BatchSize
from .digest_spec import get_test_inputs, get_test_outputs
from .sample import Sample
from .utils import VERSION


def enable_determinism(mode: Literal["seed_only", "full"]):
    """Seed and configure ML frameworks for maximum reproducibility.
    May degrade performance. Only recommended for testing reproducibility!

    Seed any random generators and (if **mode**=="full") request ML frameworks to use
    deterministic algorithms.
    Notes:
        - **mode** == "full"  might degrade performance and throw exceptions.
        - Subsequent inference calls might still differ. Call before each function
          (sequence) that is expected to be reproducible.
        - Degraded performance: Use for testing reproducibility only!
        - Recipes:
            - [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
            - [Keras](https://keras.io/examples/keras_recipes/reproducibility_recipes/)
            - [NumPy](https://numpy.org/doc/2.0/reference/random/generated/numpy.random.seed.html)
    """
    try:
        try:
            import numpy.random
        except ImportError:
            pass
        else:
            numpy.random.seed(0)
    except Exception as e:
        logger.debug(str(e))

    try:
        try:
            import torch
        except ImportError:
            pass
        else:
            _ = torch.manual_seed(0)
            torch.use_deterministic_algorithms(mode == "full")
    except Exception as e:
        logger.debug(str(e))

    try:
        try:
            import keras
        except ImportError:
            pass
        else:
            keras.utils.set_random_seed(0)
    except Exception as e:
        logger.debug(str(e))

    try:
        try:
            import tensorflow as tf  # pyright: ignore[reportMissingImports]
        except ImportError:
            pass
        else:
            tf.random.seed(0)
            if mode == "full":
                tf.config.experimental.enable_op_determinism()
            # TODO: find possibility to switch it off again??
    except Exception as e:
        logger.debug(str(e))


def test_model(
    source: Union[v0_5.ModelDescr, PermissiveFileSource],
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[List[str]] = None,
    absolute_tolerance: float = 1.5e-4,
    relative_tolerance: float = 1e-4,
    decimal: Optional[int] = None,
) -> ValidationSummary:
    """Test model inference"""
    # NOTE: `decimal` is a legacy argument and is handled in `_test_model_inference`
    return test_description(
        source,
        weight_format=weight_format,
        devices=devices,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        decimal=decimal,
        expected_type="model",
    )


def test_description(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[Literal["discover", "latest"], str] = "discover",
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    absolute_tolerance: float = 1.5e-4,
    relative_tolerance: float = 1e-4,
    decimal: Optional[int] = None,
    expected_type: Optional[str] = None,
) -> ValidationSummary:
    """Test a bioimage.io resource dynamically, e.g. prediction of test tensors for models"""
    # NOTE: `decimal` is a legacy argument and is handled in `_test_model_inference`
    rd = load_description_and_test(
        source,
        format_version=format_version,
        weight_format=weight_format,
        devices=devices,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        decimal=decimal,
        expected_type=expected_type,
    )
    return rd.validation_summary


def load_description_and_test(
    source: Union[ResourceDescr, PermissiveFileSource, BioimageioYamlContent],
    *,
    format_version: Union[Literal["discover", "latest"], str] = "discover",
    weight_format: Optional[WeightsFormat] = None,
    devices: Optional[Sequence[str]] = None,
    absolute_tolerance: float = 1.5e-4,
    relative_tolerance: float = 1e-4,
    decimal: Optional[int] = None,
    expected_type: Optional[str] = None,
) -> Union[ResourceDescr, InvalidDescr]:
    """Test RDF dynamically, e.g. model inference of test inputs"""
    # NOTE: `decimal` is a legacy argument and is handled in `_test_model_inference`
    if (
        isinstance(source, ResourceDescrBase)
        and format_version != "discover"
        and source.format_version != format_version
    ):
        warnings.warn(
            f"deserializing source to ensure we validate and test using format {format_version}"
        )
        source = dump_description(source)

    if isinstance(source, ResourceDescrBase):
        rd = source
    elif isinstance(source, dict):
        rd = build_description(source, format_version=format_version)
    else:
        rd = load_description(source, format_version=format_version)

    rd.validation_summary.env.add(
        InstalledPackage(name="bioimageio.core", version=VERSION)
    )

    if expected_type is not None:
        _test_expected_resource_type(rd, expected_type)

    if isinstance(rd, (v0_4.ModelDescr, v0_5.ModelDescr)):
        if weight_format is None:
            weight_formats: List[WeightsFormat] = [
                w for w, we in rd.weights if we is not None
            ]  # pyright: ignore[reportAssignmentType]
        else:
            weight_formats = [weight_format]
        for w in weight_formats:
            _test_model_inference(
                rd, w, devices, absolute_tolerance, relative_tolerance, decimal
            )
            if not isinstance(rd, v0_4.ModelDescr):
                _test_model_inference_parametrized(rd, w, devices)

    # TODO: add execution of jupyter notebooks
    # TODO: add more tests

    return rd


def _test_model_inference(
    model: Union[v0_4.ModelDescr, v0_5.ModelDescr],
    weight_format: WeightsFormat,
    devices: Optional[Sequence[str]],
    absolute_tolerance: float,
    relative_tolerance: float,
    decimal: Optional[int],
) -> None:
    test_name = f"Reproduce test outputs from test inputs ({weight_format})"
    logger.info("starting '{}'", test_name)
    error: Optional[str] = None
    tb: List[str] = []

    precision_args = _handle_legacy_precision_args(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        decimal=decimal,
    )

    try:
        inputs = get_test_inputs(model)
        expected = get_test_outputs(model)

        with create_prediction_pipeline(
            bioimageio_model=model, devices=devices, weight_format=weight_format
        ) as prediction_pipeline:
            results = prediction_pipeline.predict_sample_without_blocking(inputs)

        if len(results.members) != len(expected.members):
            error = f"Expected {len(expected.members)} outputs, but got {len(results.members)}"

        else:
            for m, exp in expected.members.items():
                res = results.members.get(m)
                if res is None:
                    error = "Output tensors for test case may not be None"
                    break
                try:
                    np.testing.assert_allclose(
                        res.data,
                        exp.data,
                        rtol=precision_args["relative_tolerance"],
                        atol=precision_args["absolute_tolerance"],
                    )
                except AssertionError as e:
                    error = f"Output and expected output disagree:\n {e}"
                    break
    except Exception as e:
        error = str(e)
        tb = traceback.format_tb(e.__traceback__)

    model.validation_summary.add_detail(
        ValidationDetail(
            name=test_name,
            loc=("weights", weight_format),
            status="passed" if error is None else "failed",
            recommended_env=get_conda_env(entry=dict(model.weights)[weight_format]),
            errors=(
                []
                if error is None
                else [
                    ErrorEntry(
                        loc=("weights", weight_format),
                        msg=error,
                        type="bioimageio.core",
                        traceback=tb,
                    )
                ]
            ),
        )
    )


def _test_model_inference_parametrized(
    model: v0_5.ModelDescr,
    weight_format: WeightsFormat,
    devices: Optional[Sequence[str]],
) -> None:
    if not any(
        isinstance(a.size, v0_5.ParameterizedSize)
        for ipt in model.inputs
        for a in ipt.axes
    ):
        # no parameterized sizes => set n=0
        ns: Set[v0_5.ParameterizedSize_N] = {0}
    else:
        ns = {0, 1, 2}

    given_batch_sizes = {
        a.size
        for ipt in model.inputs
        for a in ipt.axes
        if isinstance(a, v0_5.BatchAxis)
    }
    if given_batch_sizes:
        batch_sizes = {gbs for gbs in given_batch_sizes if gbs is not None}
        if not batch_sizes:
            # only arbitrary batch sizes
            batch_sizes = {1, 2}
    else:
        # no batch axis
        batch_sizes = {1}

    test_cases: Set[Tuple[v0_5.ParameterizedSize_N, BatchSize]] = {
        (n, b) for n, b in product(sorted(ns), sorted(batch_sizes))
    }
    logger.info(
        "Testing inference with {} different input tensor sizes", len(test_cases)
    )

    def generate_test_cases():
        tested: Set[Hashable] = set()

        def get_ns(n: int):
            return {
                (t.id, a.id): n
                for t in model.inputs
                for a in t.axes
                if isinstance(a.size, v0_5.ParameterizedSize)
            }

        for n, batch_size in sorted(test_cases):
            input_target_sizes, expected_output_sizes = model.get_axis_sizes(
                get_ns(n), batch_size=batch_size
            )
            hashable_target_size = tuple(
                (k, input_target_sizes[k]) for k in sorted(input_target_sizes)
            )
            if hashable_target_size in tested:
                continue
            else:
                tested.add(hashable_target_size)

            resized_test_inputs = Sample(
                members={
                    t.id: test_inputs.members[t.id].resize_to(
                        {
                            aid: s
                            for (tid, aid), s in input_target_sizes.items()
                            if tid == t.id
                        },
                    )
                    for t in model.inputs
                },
                stat=test_inputs.stat,
                id=test_inputs.id,
            )
            expected_output_shapes = {
                t.id: {
                    aid: s
                    for (tid, aid), s in expected_output_sizes.items()
                    if tid == t.id
                }
                for t in model.outputs
            }
            yield n, batch_size, resized_test_inputs, expected_output_shapes

    try:
        test_inputs = get_test_inputs(model)

        with create_prediction_pipeline(
            bioimageio_model=model, devices=devices, weight_format=weight_format
        ) as prediction_pipeline:
            for n, batch_size, inputs, exptected_output_shape in generate_test_cases():
                error: Optional[str] = None
                result = prediction_pipeline.predict_sample_without_blocking(inputs)
                if len(result.members) != len(exptected_output_shape):
                    error = (
                        f"Expected {len(exptected_output_shape)} outputs,"
                        + f" but got {len(result.members)}"
                    )

                else:
                    for m, exp in exptected_output_shape.items():
                        res = result.members.get(m)
                        if res is None:
                            error = "Output tensors may not be None for test case"
                            break

                        diff: Dict[AxisId, int] = {}
                        for a, s in res.sizes.items():
                            if isinstance((e_aid := exp[AxisId(a)]), int):
                                if s != e_aid:
                                    diff[AxisId(a)] = s
                            elif (
                                s < e_aid.min or e_aid.max is not None and s > e_aid.max
                            ):
                                diff[AxisId(a)] = s
                        if diff:
                            error = (
                                f"(n={n}) Expected output shape {exp},"
                                + f" but got {res.sizes} (diff: {diff})"
                            )
                            break

                model.validation_summary.add_detail(
                    ValidationDetail(
                        name=f"Run {weight_format} inference for inputs with"
                        + f" batch_size: {batch_size} and size parameter n: {n}",
                        loc=("weights", weight_format),
                        status="passed" if error is None else "failed",
                        errors=(
                            []
                            if error is None
                            else [
                                ErrorEntry(
                                    loc=("weights", weight_format),
                                    msg=error,
                                    type="bioimageio.core",
                                )
                            ]
                        ),
                    )
                )
    except Exception as e:
        error = str(e)
        tb = traceback.format_tb(e.__traceback__)
        model.validation_summary.add_detail(
            ValidationDetail(
                name=f"Run {weight_format} inference for parametrized inputs",
                status="failed",
                loc=("weights", weight_format),
                errors=[
                    ErrorEntry(
                        loc=("weights", weight_format),
                        msg=error,
                        type="bioimageio.core",
                        traceback=tb,
                    )
                ],
            )
        )


def _test_expected_resource_type(
    rd: Union[InvalidDescr, ResourceDescr], expected_type: str
):
    has_expected_type = rd.type == expected_type
    rd.validation_summary.details.append(
        ValidationDetail(
            name="Has expected resource type",
            status="passed" if has_expected_type else "failed",
            loc=("type",),
            errors=(
                []
                if has_expected_type
                else [
                    ErrorEntry(
                        loc=("type",),
                        type="type",
                        msg=f"expected type {expected_type}, found {rd.type}",
                    )
                ]
            ),
        )
    )


def _handle_legacy_precision_args(
    absolute_tolerance: float, relative_tolerance: float, decimal: Optional[int]
) -> Dict[str, float]:
    """
    Transform the precision arguments to conform with the current implementation.

    If the deprecated `decimal` argument is used it overrides the new behaviour with
    the old behaviour.
    """
    # Already conforms with current implementation
    if decimal is None:
        return {
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        }
    else:
        warnings.warn(
            "The argument `decimal` has been depricated in favour of "
            + "`relative_tolerance` and `absolute_tolerance`, with different validation "
            + "logic, using `numpy.testing.assert_allclose, see "
            + "'https://numpy.org/doc/stable/reference/generated/"
            + "numpy.testing.assert_allclose.html'. Passing a value for `decimal` will "
            + "cause validation to revert to the old behaviour."
        )

    # decimal overrides new behaviour,
    #   have to convert the params to emulate old behaviour
    return {
        "absolute_tolerance": 1.5 * 10 ** (-decimal),
        "relative_tolerance": 0,
    }


# TODO: Implement `debug_model()`
# def debug_model(
#     model_rdf: Union[RawResourceDescr, ResourceDescr, URI, Path, str],
#     *,
#     weight_format: Optional[WeightsFormat] = None,
#     devices: Optional[List[str]] = None,
# ):
#     """Run the model test and return dict with inputs, results, expected results and intermediates.

#     Returns dict with tensors "inputs", "inputs_processed", "outputs_raw", "outputs", "expected" and "diff".
#     """
#     inputs_raw: Optional = None
#     inputs_processed: Optional = None
#     outputs_raw: Optional = None
#     outputs: Optional = None
#     expected: Optional = None
#     diff: Optional = None

#     model = load_description(
#         model_rdf, weights_priority_order=None if weight_format is None else [weight_format]
#     )
#     if not isinstance(model, Model):
#         raise ValueError(f"Not a bioimageio.model: {model_rdf}")

#     prediction_pipeline = create_prediction_pipeline(
#         bioimageio_model=model, devices=devices, weight_format=weight_format
#     )
#     inputs = [
#         xr.DataArray(load_array(str(in_path)), dims=input_spec.axes)
#         for in_path, input_spec in zip(model.test_inputs, model.inputs)
#     ]
#     input_dict = {input_spec.name: input for input_spec, input in zip(model.inputs, inputs)}

#     # keep track of the non-processed inputs
#     inputs_raw = [deepcopy(input) for input in inputs]

#     computed_measures = {}

#     prediction_pipeline.apply_preprocessing(input_dict, computed_measures)
#     inputs_processed = list(input_dict.values())
#     outputs_raw = prediction_pipeline.predict(*inputs_processed)
#     output_dict = {output_spec.name: deepcopy(output) for output_spec, output in zip(model.outputs, outputs_raw)}
#     prediction_pipeline.apply_postprocessing(output_dict, computed_measures)
#     outputs = list(output_dict.values())

#     if isinstance(outputs, (np.ndarray, xr.DataArray)):
#         outputs = [outputs]

#     expected = [
#         xr.DataArray(load_array(str(out_path)), dims=output_spec.axes)
#         for out_path, output_spec in zip(model.test_outputs, model.outputs)
#     ]
#     if len(outputs) != len(expected):
#         error = f"Number of outputs and number of expected outputs disagree: {len(outputs)} != {len(expected)}"
#         print(error)
#     else:
#         diff = []
#         for res, exp in zip(outputs, expected):
#             diff.append(res - exp)

#     return {
#         "inputs": inputs_raw,
#         "inputs_processed": inputs_processed,
#         "outputs_raw": outputs_raw,
#         "outputs": outputs,
#         "expected": expected,
#         "diff": diff,
#     }
