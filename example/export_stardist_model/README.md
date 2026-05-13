Here are the scripts used to create a bioimage.io export of a stardist model.

```console
python bioimageio_export.py
cd output && unzip stardist_bioimageio_2D_versatile_fluo.zip test_output_instances.npy
bioimageio test stardist_bioimageio_2D_versatile_fluo.zip --working-dir=export_test
cd ..
python analyze_export.py
````
