cd "$(dirname "$0")"  # cd to folder this script is in

# patch pydantic to hide pydantic attributes that somehow show up in the docs
# (not even as inherited, but as if the documented class itself would define them)
pydantic_main=$(python -c "import pydantic;from pathlib import Path;print(Path(pydantic.__file__).parent / 'main.py')")

patch --verbose --forward -p1 $pydantic_main < mark_pydantic_attrs_private.patch

cd ../..  # cd to repo root
pdoc \
    --docformat google \
    --logo "https://bioimage.io/static/img/bioimage-io-logo.svg" \
    --logo-link "https://bioimage.io/" \
    --favicon "https://bioimage.io/static/img/bioimage-io-icon-small.svg" \
    --footer-text "bioimageio.core $(python -c 'import bioimageio.core;print(bioimageio.core.__version__)')" \
    -o ./dist bioimageio.core bioimageio.spec  # generate bioimageio.spec as well for references
