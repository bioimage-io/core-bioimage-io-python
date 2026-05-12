Here are the scripts used to create a bioimage.io export of the Cellpose-SAM model.

1. `cellpose_original.py` Run original cellpose model and save an analog input and output for bioimageio tests
1. `bioimageio_export` Create a bioimage.io model description, then export and test it.
1. `analyse_export.py` Compare cellpose results to bioimage.io results in depth.

```console
python cellpose_original.py
python bioimageio_export.py
python analyse_export.py
````
