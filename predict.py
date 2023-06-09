# 3deify - Copyright (C) Joseph M. Shunia, 2023
import cv2
import numpy as np
import os
import tensorflow as tf
import trimesh
import trimesh.voxel
import sys

IMAGE_RES = 128
INPUT_IMAGE_PATH = '_predict/input.png'
if len(sys.argv) >= 2:
    INPUT_IMAGE_PATH = sys.argv[1]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_RES, IMAGE_RES))
    img = img / 255.0
    return img[np.newaxis, :, :, :]

def postprocess_voxels(voxel_data, threshold=0.5):
    return (voxel_data > threshold).astype(np.uint8)

def voxels_to_mesh(voxels):
    volume = trimesh.voxel.VoxelGrid(voxels)
    mesh = volume.marching_cubes
    return mesh

def save_mesh_to_obj(mesh, output_path):
    mesh.export(output_path)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Preprocess input image

image_name = os.path.basename(INPUT_IMAGE_PATH)
image_name_without_ext = os.path.splitext(image_name)[0]
preprocessed_image = preprocess_image(INPUT_IMAGE_PATH)

# Predict the 3D voxel data
predicted_voxels = model.predict(preprocessed_image)[0]

# Postprocess the voxel data
threshold = 0.0
processed_voxels = postprocess_voxels(predicted_voxels, threshold)

# Convert voxels to a mesh
mesh = voxels_to_mesh(processed_voxels)

# Save the mesh as an OBJ file
output_path = f'_predict/{image_name}.obj'
save_mesh_to_obj(mesh, output_path)
