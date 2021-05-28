from pathlib import Path
from setuptools import find_namespace_packages, setup

# Get the long description from the README file
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="bioimageio",
    version="0.3b",
    description="Parser library for bioimage model zoo specs",
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
    install_requires=[
        "PyYAML>=5.2",
        "bioimageio.spec @ git+http://github.com/bioimage-io/spec-bioimage-io#egg=bioimageio.spec",
        "dataclasses; python_version>='3.7.2,<3.9'",
        "imageio>=2.5",
        "marshmallow>=3.3.0,<3.5",
        "marshmallow_jsonschema",
        "marshmallow_union",
        "requests",
        "ruamel.yaml",
        "typer",
        "typing-extensions",
    ],
    extras_require={"core": ["numpy", "sklearn", "imageio"], "test": ["pytest", "tox"], "dev": ["pre-commit"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/python-bioimage-io/issues",
        "Source": "https://github.com/bioimage-io/python-bioimage-io",
    },
)
