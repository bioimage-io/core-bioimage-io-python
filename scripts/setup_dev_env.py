# untested draft!
import subprocess
from os import chdir
from pathlib import Path


def run(prompt: str):
    _ = subprocess.run(prompt, check=True, capture_output=True)


if __name__ == "__main__":
    repo_dir = Path(__file__).parent.parent.parent
    cur_dir = Path().resolve()
    chdir(str(repo_dir))
    try:
        run("mamba env create --file core-bioimage-io/dev/env.yaml")
        run(
            "pip install --no-deps --config-settings editable_mode=compat -e spec-bioimage-io"
        )
        run(
            "pip install --no-deps --config-settings editable_mode=compat -e core-bioimage-io"
        )
    except Exception:
        chdir(cur_dir)
        raise
