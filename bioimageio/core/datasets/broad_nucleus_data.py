import collections
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Union
from urllib.request import urlretrieve

import imageio
import numpy
import numpy as np

from bioimageio.core.datasets.base import Dataset
from bioimageio.spec.nodes import Axes, OutputTensor

BIOIMAGEIO_CACHE_PATH = Path(os.getenv("BIOIMAGEIO_CACHE_PATH", Path.home() / "bioimageio_cache"))

try:
    from typing import OrderedDict
except ImportError:
    from typing import MutableMapping as OrderedDict


def download_data(url, data_dir: Path, prefix: str):
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(data_dir / "tmp.zip")

    # retrieve url
    urlretrieve(url, tmp)

    # extract zips to target folder
    with zipfile.ZipFile(tmp, "r") as f:
        for ff in f.namelist():
            if ff.startswith(prefix):
                f.extract(ff, data_dir)
    os.remove(tmp)


def load_file_list(file_list, data_path: Path, is_tif=False):
    files = []
    with open(file_list, "r") as f:
        for ll in f:
            path = data_path / ll.strip("\n")
            if is_tif:
                path = path.with_suffix(".tif")
            assert path.exists(), path
            files.append(path)
    files.sort()
    return files


def load_images(files):
    images = []
    for ff in files:
        im = np.asarray(imageio.imread(ff))
        # the labels have multiple channels, but we only care for the
        # first of it
        if im.ndim == 3:
            im = im[..., 0]
        images.append(im)

    return np.stack(images)


class BroadNucleusDataBinarized(Dataset):
    # TODO store hashes and validate
    urls = {
        "images": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        "masks": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "metadata": "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    }

    def get_data(self, data_dir: Path, subset: str):
        assert subset in ["training", "validation", "test"]
        for prefix, url in self.urls.items():
            if not (data_dir / prefix).exists():
                print("Downloading", prefix, "...")
                download_data(url, data_dir, prefix + "/")

        train_list = data_dir / "metadata" / f"{subset}.txt"
        label_list = load_file_list(train_list, data_dir / "masks")
        labels = load_images(label_list)

        # we binarize the labels
        labels = labels.astype("bool")

        image_list = load_file_list(train_list, data_dir / "images", is_tif=True)
        images = load_images(image_list).astype("uint16", copy=False)

        # crop = np.s_[:, :512, :512]
        # images = images[crop]
        # labels = labels[crop]
        assert images.shape == labels.shape

        return images, labels

    def __init__(self, subset: str = "training", **super_kwargs):
        self.x, self.y = self.get_data(BIOIMAGEIO_CACHE_PATH / "BroadNucleusDataBinarizedPyBioReader", subset)
        if len(self.x) != len(self.y):
            raise RuntimeError("Invalid data")

        assert len(self.x.shape) == 3
        assert len(self.y.shape) == 3

        outputs = [
            OutputTensor(
                name="raw",
                axes=Axes("byx"),
                data_type="uint16",
                data_range=(numpy.iinfo(numpy.uint16).min, numpy.iinfo(numpy.uint16).max),
                shape=list(self.x.shape),
                halo=[0, 0, 0],
                description="raw",
                postprocessing=[],
            ),
            OutputTensor(
                name="target",
                axes=Axes("byx"),
                data_type="float32",
                data_range=(numpy.float("-inf"), numpy.float("inf")),
                shape=list(self.y.shape),
                halo=[0, 0, 0],
                description="target",
                postprocessing=[],
            ),
        ]

        super().__init__(outputs=outputs, **super_kwargs)

    def __getitem__(
        self, index: Union[Tuple[slice, slice, slice], Dict[str, Tuple[slice, slice, slice]]]
    ) -> OrderedDict[str, numpy.ndarray]:
        if isinstance(index, tuple):
            index = {d: index for d in "xy"}

        batch = collections.OrderedDict(x=self.x[index["x"]], y=self.y[index["y"]])
        self.apply_transformation(batch)
        return batch
