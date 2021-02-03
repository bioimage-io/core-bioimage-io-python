from pathlib import Path
from pprint import pprint

import requests
import typer
from marshmallow import ValidationError
from ruamel.yaml import YAML

from pybio.spec import schema

yaml = YAML(typ="safe")

app = typer.Typer()  # https://typer.tiangolo.com/


@app.command()
def verify_spec(model_yaml: Path):
    try:
        spec_data = yaml.load(model_yaml)
    except Exception as e:
        pprint(e)
        code = 1
    else:
        try:
            verify_model_data(spec_data)
        except ValidationError as e:
            pprint(e.messages)
            code = 1
        else:
            code = 0

    raise typer.Exit(code=code)


def verify_model_data(model_data: dict):
    schema.Model().load(model_data)


def verify_bioimageio_manifest_data(manifest_data: dict):
    manifest = schema.BioImageIoManifest().load(manifest_data)

    code = 0
    for model in manifest["model"]:
        try:
            response = requests.get(model["source"], stream=True)
            model_data = yaml.load(response.content)
            verify_model_data(model_data)
        except ValidationError as e:
            print("invalid model:", model["source"])
            pprint(e.messages)
            code = 1
        except Exception as e:
            print("invalid model:", model["source"])
            pprint(e)
            code = 1

    return code


@app.command()
def verify_bioimageio_manifest(manifest_yaml: Path):
    try:
        manifest_data = yaml.load(manifest_yaml)
    except Exception as e:
        print("invalid manifest", manifest_yaml)
        pprint(e)
        code = 1
    else:
        code = verify_bioimageio_manifest_data(manifest_data)

    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
