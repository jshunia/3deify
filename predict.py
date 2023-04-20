import cv2
import numpy as np
import tensorflow as tf
import trimesh
import trimesh.voxel

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
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
input_image_path = '_output/cube.obj_45_45.png'
preprocessed_image = preprocess_image(input_image_path)

# Predict the 3D voxel data
predicted_voxels = model.predict(preprocessed_image)[0]

# Postprocess the voxel data
threshold = 0.5
processed_voxels = postprocess_voxels(predicted_voxels, threshold)

# Convert voxels to a mesh
mesh = voxels_to_mesh(processed_voxels)

# Save the mesh as an OBJ file
output_path = f'_predict/model.obj'
save_mesh_to_obj(mesh, output_path)
