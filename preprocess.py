# 3deify - Copyright (C) Joseph M. Shunia, 2023
import cv2
import glob
import math
import numpy as np
import os
import re
import sys
import tensorflow as tf
import trimesh
from trimesh.voxel import creation
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

OBJ_FILE_FILTER = '*.obj'
if len(sys.argv) >= 2:
    OBJ_FILE_FILTER = sys.argv[1]

print(f'Preprocessing using object file filter: {OBJ_FILE_FILTER}')

OBJ_DIR = '_objects'
IMAGE_DIR = '_images'
VOXEL_RES = 32
IMAGE_RES = 128

def get_obj_trimesh(obj):
    output_file_path = "temp.obj"  # Replace with the desired output file path

    # Make sure the object type is MESH
    if obj.type == "MESH":
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Select only the target object
        obj.select_set(True)

        # Set the target object as the active object
        bpy.context.view_layer.objects.active = obj

        # Export the object to .obj file
        bpy.ops.export_scene.obj(
            filepath=output_file_path,
            use_selection=True,  # Export only the selected object
            use_mesh_modifiers=True,  # Apply mesh modifiers
            axis_forward='-Z',  # Forward axis
            axis_up='Y'  # Up axis
        )
        
        mesh = trimesh.load_mesh(output_file_path)
        return mesh
    else:
        print("Error: 'obj' is not a mesh object.")
        exit(1)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_RES, IMAGE_RES))
    img = img / 255.0
    return img

# Loads the 3d object and converts it to a format that can be used as training data.
def preprocess_3d_object(obj_path, voxel_resolution=VOXEL_RES):
    mesh = trimesh.load_mesh(obj_path)
    pitch = mesh.extents.max() / voxel_resolution
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)

    # Clip the dimensions of the voxel matrix to the maximum allowed voxel_resolution
    clipped_voxel_matrix = voxel_matrix[:voxel_resolution, :voxel_resolution, :voxel_resolution]

    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:clipped_voxel_matrix.shape[0], :clipped_voxel_matrix.shape[1], :clipped_voxel_matrix.shape[2]] = clipped_voxel_matrix

    return padded_voxel_matrix
	
# Loads the 3d object, rotates it, and converts it to a format that can be used as training data.
def preprocess_3d_object_rotate(obj_path, angle_x, angle_y, voxel_resolution=VOXEL_RES):
    # Load the 3D object
    mesh = trimesh.load_mesh(obj_path)
	
    # Rotate the object
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_x, [1,0,0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_y, [0,1,0]))
    
    #mesh.rotation_euler[0] = math.radians(angle_x)
    #mesh.rotation_euler[1] = math.radians(angle_y)
    #mesh.rotation_euler[2] = 0
    #mesh.apply_transform(trimesh.transformations.rotation_matrix(rot_angle, rot_axis))
	
    pitch = mesh.extents.max() / voxel_resolution
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)

    # Clip the dimensions of the voxel matrix to the maximum allowed voxel_resolution
    clipped_voxel_matrix = voxel_matrix[:voxel_resolution, :voxel_resolution, :voxel_resolution]

    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:clipped_voxel_matrix.shape[0], :clipped_voxel_matrix.shape[1], :clipped_voxel_matrix.shape[2]] = clipped_voxel_matrix

    return padded_voxel_matrix
   
# Loads the 3d object, rotates it randomly, translates it randomly, and converts it to a format that can be used as training data.
# In theory, this preprocessing MAY help train the model to better recognize multiple objects within a scene (TBD).
def preprocess_3d_object_random(obj_path, voxel_resolution=VOXEL_RES):
    # Load the 3D object
    mesh = trimesh.load_mesh(obj_path)

    # Generate a random rotation angle and axis
    rot_axis = np.random.rand(3)
    rot_angle = np.random.rand() * 2 * np.pi

    # Rotate the object
    mesh.apply_transform(trimesh.transformations.rotation_matrix(rot_angle, rot_axis))

    # Get the dimensions of the object
    min_bounds, max_bounds = mesh.bounds
    dims = max_bounds - min_bounds

    # Generate a random translation vector
    tx = np.random.uniform(-dims[0], dims[0])
    ty = np.random.uniform(-dims[1], dims[1])
    tz = np.random.uniform(-dims[2], dims[2])

    # Translate the object
    mesh.apply_translation([tx, ty, tz])
	
	# Scale the object by 0.5
    mesh.apply_scale(0.5)

    # Compute the pitch for voxelization
    pitch = mesh.extents.max() / voxel_resolution

    # Voxelize the object
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)
    
    # Clip the dimensions of the voxel matrix to the maximum allowed voxel_resolution
    clipped_voxel_matrix = voxel_matrix[:voxel_resolution, :voxel_resolution, :voxel_resolution]

    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:clipped_voxel_matrix.shape[0], :clipped_voxel_matrix.shape[1], :clipped_voxel_matrix.shape[2]] = clipped_voxel_matrix

    return padded_voxel_matrix

def parse_rot_x_y(filename):
    parts = filename.split('_')
    if len(parts) >= 2:
        return int(parts[-2]), int(parts[-1])
    else:
        raise ValueError('The input string must have at least two underscore-separated values.')

# Load your dataset
image_data = []
obj_data = []

obj_paths = glob.glob(os.path.join(OBJ_DIR, OBJ_FILE_FILTER))

for obj_path in obj_paths:
    # Get the 3D object name without the extension
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]

    # Find all images that start with the object name
    image_paths = glob.glob(os.path.join(IMAGE_DIR, f"{obj_name}*.png"))
    
    # Preprocess 3D object
    obj = preprocess_3d_object(obj_path)

    # Preprocess and associate each 2D image with the corresponding 3D object
    images = [preprocess_image(image_path) for image_path in image_paths]
    i = 0
    for image in images:
        image_path = image_paths[i]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        image_data.append(image)
        angle_x, angle_y = parse_rot_x_y(image_name)
        obj1 = preprocess_3d_object_rotate(obj_path, angle_x, angle_y)
        obj_data.append(obj1)
        #obj_data.append(obj)
        i += 1
        
images_2d = np.array(image_data)
objs_3d = np.array(obj_data)
#objs_3d = np.array(obj_data, dtype=object)
#objs_3d = np.asarray(obj_data).astype('float32')

# Split the dataset into training and validation sets
train_images, val_images, train_objs, val_objs = train_test_split(images_2d, objs_3d, test_size=0.2, random_state=42)

# Save the datasets as NumPy files for future use
np.save('train_images.npy', train_images)
np.save('val_images.npy', val_images)
np.save('train_objs.npy', train_objs)
np.save('val_objs.npy', val_objs)

#EOF