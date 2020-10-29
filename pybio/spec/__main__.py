from pathlib import Path

import typer
from ruamel_yaml import YAML

from pybio.spec import schema


yaml = YAML(typ="safe")

app = typer.Typer()  # https://typer.tiangolo.com/


@app.command()
def verify_spec(model_yaml: Path):
    spec_data = yaml.load(model_yaml)
    schema.ModelSpec().load(spec_data)
    print('valid')


if __name__ == "__main__":
    app()
