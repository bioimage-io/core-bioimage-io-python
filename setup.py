import json
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent.resolve()
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "bioimageio" / "core" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text())["version"]


_ = setup(
    name="bioimageio.core",
    version=VERSION,
    description="Python functionality for the bioimage model zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/core-bioimage-io-python",
    author="Bioimage Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    packages=find_namespace_packages(exclude=["tests"]),
    install_requires=[
        "bioimageio.spec ==0.5.4.3",
        "h5py",
        "imagecodecs",
        "imageio>=2.10",
        "loguru",
        "numpy",
        "pydantic-settings>=2.5,<3",
        "pydantic>=2.7.0,<3",
        "ruyaml",
        "tqdm",
        "typing-extensions",
        "xarray>=2023.01,<2025.3.0",
    ],
    include_package_data=True,
    extras_require={
        "pytorch": (
            pytorch_deps := ["torch>=1.6,<3", "torchvision>=0.21", "keras>=3.0,<4"]
        ),
        "tensorflow": ["tensorflow", "keras>=2.15,<4"],
        "onnx": ["onnxruntime"],
        "tests": (test_deps := ["pytest", "pytest-cov"]),  # minimal test requirements
        "dev": (
            test_deps
            + pytorch_deps
            + [
                "black",
                "cellpose",  # for model testing
                "httpx",
                "jupyter-black",
                "jupyter",
                "matplotlib",
                "monai",  # for model testing
                "onnx",
                "onnxruntime",
                "packaging>=17.0",
                "pdoc",
                "pre-commit",
                "pyright==1.1.402",
                "segment-anything",  # for model testing
                "timm",  # for model testing
                # "crick",  # currently requires python<=3.9
            ]
        ),
    },
    project_urls={
        "Bug Reports": "https://github.com/bioimage-io/core-bioimage-io-python/issues",
        "Source": "https://github.com/bioimage-io/core-bioimage-io-python",
    },
    entry_points={"console_scripts": ["bioimageio = bioimageio.core.__main__:main"]},
)
