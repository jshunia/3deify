import cv2
import glob
import math
import numpy as np
import os
import tensorflow as tf
import trimesh
from trimesh.voxel import creation
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)
    
def preprocess_3d_model(model_path, voxel_resolution=32):
    mesh = trimesh.load_mesh(model_path)
    pitch = mesh.extents.max() / voxel_resolution
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)
    return voxel_matrix

# Load your dataset
model_paths = [ '_input/cube.obj' ]
image_directory = '_output'

image_data = []
model_data = []

for model_path in model_paths:
    # Get the 3D model name without the extension
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Find all images that start with the model name
    image_paths = glob.glob(os.path.join(image_directory, f"{model_name}*.png"))

    # Preprocess 3D model
    model = preprocess_3d_model(model_path)

    # Preprocess and associate each 2D image with the corresponding 3D model
    for image_path in image_paths:
        image = preprocess_image(image_path)
        image_data.append(image)
        model_data.append(model)

images_2d = np.concatenate(image_data, axis=0)
models_3d = np.array(model_data)

# Split the dataset into training and validation sets
train_images, val_images, train_models, val_models = train_test_split(images_2d, models_3d, test_size=0.2, random_state=42)

# Save the datasets as NumPy files for future use
np.save('train_images.npy', train_images)
np.save('val_images.npy', val_images)
np.save('train_models.npy', train_models)
np.save('val_models.npy', val_models)

#EOF