import json
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent.resolve()
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "bioimageio" / "core" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text())["version"]


setup(
    name="bioimageio.core",
    version=VERSION,
    description="Python functionality for the bioimage model zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/core-bioimage-io-python",
    author="Bioimage Team",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_namespace_packages(exclude=["tests"]),  # Required
    install_requires=[
        "bioimageio.spec==0.4.9.*",
        "imageio>=2.5",
        "numpy",
        "ruamel.yaml",
        "tqdm",
        "xarray",
        "tifffile",
    ],
    include_package_data=True,
    extras_require={
        "test": ["pytest", "black", "mypy"],
        "dev": ["pre-commit"],
        "pytorch": ["torch>=1.6", "torchvision"],
        "tensorflow": ["tensorflow"],
        "onnx": ["onnxruntime"],
    },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/core-bioimage-io-python/issues",
        "Source": "https://github.com/bioimage-io/core-bioimage-io-python",
    },
    entry_points={"console_scripts": ["bioimageio = bioimageio.core.__main__:app"]},
)
