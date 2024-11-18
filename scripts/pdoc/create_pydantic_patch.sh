pydantic_root=$(python -c "import pydantic;from pathlib import Path;print(Path(pydantic.__file__).parent)")
main=$pydantic_root'/main.py'
original="$(dirname "$0")/original.py"
patched="$(dirname "$0")/patched.py"

if [ -e $original ]
then
    echo "found existing $original"
else
    cp --verbose $main $original
fi

if [ -e $patched ]
then
    echo "found existing $patched"
else
    cp --verbose $main $patched
    echo "Please update $patched, then press enter to continue"
    read
fi

patch_file="$(dirname "$0")/mark_pydantic_attrs_private.patch"
diff -au $original $patched > $patch_file
echo "content of $patch_file:"
cat $patch_file
