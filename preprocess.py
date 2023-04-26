# 3deify - Copyright (C) Joseph M. Shunia, 2023
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

VOXEL_RESOLUTION = 32
IMAGE_RESOLUTION = 128

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_RESOLUTION, IMAGE_RESOLUTION))
    img = img / 255.0
    return img
    
def preprocess_3d_model_standard(model_path, voxel_resolution=VOXEL_RESOLUTION):
    mesh = trimesh.load_mesh(model_path)
    pitch = mesh.extents.max() / voxel_resolution
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)

    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:voxel_matrix.shape[0], :voxel_matrix.shape[1], :voxel_matrix.shape[2]] = voxel_matrix

    return padded_voxel_matrix
	
# In theory, this preprocessing should help train the model to better recognize multiple objects within a scene.
def preprocess_3d_model(model_path, voxel_resolution=VOXEL_RESOLUTION):
    # Load the 3D model
    mesh = trimesh.load_mesh(model_path)

    # Generate a random rotation angle and axis
    rot_axis = np.random.rand(3)
    rot_angle = np.random.rand() * 2 * np.pi

    # Rotate the model
    mesh.apply_transform(trimesh.transformations.rotation_matrix(rot_angle, rot_axis))

    # Get the dimensions of the model
    min_bounds, max_bounds = mesh.bounds
    dims = max_bounds - min_bounds

    # Generate a random translation vector
    tx = np.random.uniform(-dims[0], dims[0])
    ty = np.random.uniform(-dims[1], dims[1])
    tz = np.random.uniform(-dims[2], dims[2])

    # Translate the model
    mesh.apply_translation([tx, ty, tz])
	
	# Scale the model by 0.5
    mesh.apply_scale(0.5)

    # Compute the pitch for voxelization
    pitch = mesh.extents.max() / voxel_resolution

    # Voxelize the model
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)
    
    # Clip the dimensions of the voxel matrix to the maximum allowed voxel_resolution
    clipped_voxel_matrix = voxel_matrix[:voxel_resolution, :voxel_resolution, :voxel_resolution]

    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:clipped_voxel_matrix.shape[0], :clipped_voxel_matrix.shape[1], :clipped_voxel_matrix.shape[2]] = clipped_voxel_matrix

    return padded_voxel_matrix

# Load your dataset
model_paths = [ '_input/cube.obj', '_input/ball.obj', '_input/capsule.obj'  ]
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
    images = [preprocess_image(image_path) for image_path in image_paths]
    for image in images:
        image_data.append(image)
        model1 = preprocess_3d_model(model_path)
        model_data.append(model1)
        #model_data.append(model)

images_2d = np.array(image_data)
models_3d = np.array(model_data)
#models_3d = np.array(model_data, dtype=object)
#models_3d = np.asarray(model_data).astype('float32')

# Split the dataset into training and validation sets
train_images, val_images, train_models, val_models = train_test_split(images_2d, models_3d, test_size=0.2, random_state=42)

# Save the datasets as NumPy files for future use
np.save('train_images.npy', train_images)
np.save('val_images.npy', val_images)
np.save('train_models.npy', train_models)
np.save('val_models.npy', val_models)

#EOF