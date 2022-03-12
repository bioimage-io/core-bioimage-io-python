from typing import Dict, Set, TYPE_CHECKING


if TYPE_CHECKING:
    import xarray as xr
    from bioimageio.core.statistical_measures import Measure, MeasureValue

try:
    from typing import Literal
except ImportError:
    from typing import Literal  # type: ignore

TensorName = str
FixedMode = Literal["fixed"]
SampleMode = Literal["per_sample"]
DatasetMode = Literal["per_dataset"]
Mode = Literal[FixedMode, SampleMode, DatasetMode]

FIXED: FixedMode = "fixed"
PER_SAMPLE: SampleMode = "per_sample"
PER_DATASET: DatasetMode = "per_dataset"
MODES: Set[Mode] = {FIXED, PER_SAMPLE, PER_DATASET}

Sample = Dict[TensorName, xr.DataArray]
RequiredMeasures = Dict[Literal[SampleMode, DatasetMode], Dict[TensorName, Set[Measure]]]
ComputedMeasures = Dict[Literal[SampleMode, DatasetMode], Dict[TensorName, Dict[Measure, MeasureValue]]]
