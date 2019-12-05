import os
import zipfile
from typing import Tuple
from urllib.request import urlretrieve

import numpy
import numpy as np
import imageio

from pybio.readers.base import PyBioReader


def download_data(url, data_dir, prefix):
    os.makedirs(data_dir, exist_ok=True)
    tmp = os.path.join(data_dir, "tmp.zip")

    # retrieve url
    urlretrieve(url, tmp)

    # extract zips to target folder
    with zipfile.ZipFile(tmp, "r") as f:
        for ff in f.namelist():
            if ff.startswith(prefix):
                f.extract(ff, data_dir)
    os.remove(tmp)


def load_file_list(file_list, data_folder, is_tif=False):
    files = []
    with open(file_list, "r") as f:
        for ll in f:
            path = os.path.join(data_folder, ll).strip("\n")
            if is_tif:
                path, _ = os.path.splitext(path)
                path += ".tif"
            assert os.path.exists(path), path
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


class BroadNucleusData(PyBioReader):
    # TODO store hashes and validate
    urls = {
        "images": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        "masks": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "metadata": "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    }

    def get_data(self, data_dir):
        for prefix, url in self.urls.items():
            if not os.path.exists(os.path.join(data_dir, prefix)):
                print("Downloading", prefix, "...")
                download_data(url, data_dir, prefix + "/")

        train_list = os.path.join(data_dir, "metadata", "training.txt")
        label_list = load_file_list(train_list, os.path.join(data_dir, "masks"))
        labels = load_images(label_list)

        # we binarize the labels
        labels[labels > 0] = 1
        labels = labels.astype("float32")

        image_list = load_file_list(train_list, os.path.join(data_dir, "images"), is_tif=True)
        images = load_images(image_list).astype("float32")

        crop = np.s_[:, :512, :512]
        images = images[crop]
        labels = labels[crop]
        assert images.shape == labels.shape

        return images, labels

    def __init__(self, data_dir="./tmp"):
        self.x, self.y = self.get_data(data_dir)
        if len(self.x) != len(self.y):
            raise RuntimeError("Invalid data")

        assert len(self.x.shape) == 3
        assert len(self.y.shape) == 3
        self._shape = self.x.shape, self.y.shape
        self._axes = "zyx", "zyx"  # todo: check axes
        super().__init__()


    def __getitem__(self, index: Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        x, y = self.x[index[0]], self.y[index[1]]
        return x, y
