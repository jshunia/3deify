# 3deify
An experimental image-to-3d model.
This is unfinished and is a work in progress!

# Scripts

## train.py
This is used for training our model. Uses tensorflow.

## create_dataset.py
This script takes a 3D object file (.obj or other) and outputs a series of 2D images of the 3D object at varying perspectives (angles). It is working but incomplete. Eventually this will be used to create datasets for training our model.

To run `create_dataset.py`, you must first install Blender and add `blender.exe` to the `PATH` environment variable. Afterwards, you can run it from the command line using Blender, for example: `blender -b -P create_dataset.py` (note: arguments are not yet supported).

Copyright (C) Joseph Shunia, 2023
