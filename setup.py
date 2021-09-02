import json
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent.resolve()
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "bioimageio" / "spec" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text())["version"]


setup(
    name="bioimageio.core",
    version=VERSION,
    description="Python functionality for the bioimage model zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/python-bioimage-io",
    author="Bioimage Team",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_namespace_packages(exclude=["tests"]),  # Required
    install_requires=["bioimageio.spec", "imageio>=2.5", "numpy", "xarray"],
    extras_require={
        "test": ["pytest", "tox"],
        "dev": ["pre-commit"],
        "pytorch": ["pytorch>=1.6", "torchvision", "cudatoolkit>=10.1"],
        "tensorflow": ["tensorflow"],
        "onnx": ["onnxruntime"],
    },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/python-bioimage-io/issues",
        "Source": "https://github.com/bioimage-io/python-bioimage-io",
    },
    entry_points={"console_scripts": ["bioimageio = bioimageio.core.__main__:app"]},
    },
)
