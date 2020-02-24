from io import open
from os import path

from setuptools import find_namespace_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pybio.core",
    version="0.1a",
    description="Parser library for bioimage model zoo specs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/python-bioimage-io",
    author="Bioimage Team",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_namespace_packages(exclude=["tests"]),  # Required
    install_requires=["dataclasses; python_version<'3.7'", "marshmallow>=3.3.0,<3.5", "PyYAML>=5.2", "requests"],
    extras_require={"core": ["numpy", "sklearn", "imageio"], "test": ["pytest", "tox"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/python-bioimage-io/issues",
        "Source": "https://github.com/bioimage-io/python-bioimage-io",
    },
)
