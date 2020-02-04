import os
import zipfile
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import imageio
import numpy
import numpy as np

from pybio.core.readers.base import PyBioReader


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


class BroadNucleusDataBinarized(PyBioReader):
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
        images = load_images(image_list).astype("float32")

        # crop = np.s_[:, :512, :512]
        # images = images[crop]
        # labels = labels[crop]
        assert images.shape == labels.shape

        return images, labels

    def __init__(
        self,
        data_dir: Path = Path(__file__).parent / "../../../cache/BroadNucleusDataBinarized",
        subset: str = "training",
        **super_kwargs,
    ):
        self.x, self.y = self.get_data(data_dir, subset)
        if len(self.x) != len(self.y):
            raise RuntimeError("Invalid data")

        assert len(self.x.shape) == 3
        assert len(self.y.shape) == 3

        super().__init__(
            output=Path(__file__).parent / "../../../specs/readers/BroadNucleusDataBinarized.reader.yaml",
            dynamic_shape=(self.x.shape, self.y.shape),
            **super_kwargs
        )
        assert self.axes == ("bxy", "bxy"), self.axes

    def __getitem__(
        self, index: Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        x, y = self.x[index[0]], self.y[index[1]]
        return self.apply_transformations(x, y)
