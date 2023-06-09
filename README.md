# 3DEIFY
An experimental image-to-3d machine learning framework which is capable
of training models to predict a 3d object from a single 2d image using
zero-shot classification.

# Setup
1. Install [Blender (3.5.*)](https://www.blender.org/download/).
2. Install [Python (3.11.*)](https://www.python.org/downloads/)
3. Install [pip (23.1.*)](https://pypi.org/project/pip/)
4. Run the `install_dependencies.sh` script from the command line to install required packages.

# Files

## install_dependencies.sh
This script installs the following required packages:

* numpy: a Python library for working with arrays and numerical computing.
* tensorflow: an open-source machine learning framework for building and training neural networks.
* trimesh: a Python library for working with 3D models and meshes.
* opencv-python: a Python library for computer vision and image processing.
* scikit-learn: a Python library for machine learning and data analysis.

## init.py
This script creates the training dataset, creates the model, and trains it for 10 epochs.

## create_datasets.py
This script creates the 2d images for all .obj files within the `_objects` directory.

## create_dataset.py
This script takes single a 3D object file (.obj or other) and outputs a series of 2D images of the 3D object at varying perspectives (angles).This is used to create datasets for training our model.

To run `create_dataset.py`, you must first install Blender and add `blender.exe` to the `PATH` environment variable. Afterwards, you can run it from the command line using Blender, for example: `blender -b -P create_dataset.py` (note: arguments are not yet supported).

## preprocess.py
This script creates the .npy files for our dataset. Run after `create_dataset.py` and before `train.py`.

## train.py
This is used for training our model. Uses tensorflow.

## predict.py
This script takes in a 2d image from which it predicts and outputs a 3d object (.obj file).

# Usage Example
1. Follow the steps in the `Setup` section to install prerequisites and dependencies.
2. Run `init.py` to initialize the training data and model.
3. Run `predict.py` to predict the 3d object for a 2d image. This will output a `model.obj` file to the `_predict` directory.
4. Open the `_predict/model.obj` file in a 3d object viewer to view the result and see how well the model performed.

If the result is not satisfactory, run `python train.py <epochs>` to train the model further and/or add additional training data. 

OR

1. Follow the steps in the `Setup` section to install prerequisites and dependencies.
2. Add 3d objects (.obj files) to the `_input` directory. Optional. Some 3d objects are already included in this repo.
3. Run `create_dataset.py` for every 3d object within the `_input` directory using Blender Python: `blender -b -P create_dataset.py`.
4. Run `preprocess.py` to create the `.npy` files for your dataset.
5. Run `train.py` to create and train your model. This will output a `model.h5` file once training is complete. Running this script repeatedly will further train the existing model.
6. Run `predict.py` to predict the 3d object for a 2d image. This will output a `model.obj` file to the `_predict` directory.
7. Open the `_predict/model.obj` file in a 3d object viewer to view the result and see how well the model performed.

# Notes
This source code in this repository is a work in progress.

Copyright (C) Joseph Shunia, 2023
Last Updated: April 26, 2023
