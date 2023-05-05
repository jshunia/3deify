# 3deify - Copyright (C) Joseph M. Shunia, 2023
# NOTE: To run this script, save it to a file (e.g., create_dataset.py) and run it from the command line using Blender's Python: blender -b -P init_dataset.py <object_name_filter>
import bpy
import cv2
import glob
import os
import numpy as np
import math
import mathutils
import sys
import trimesh
from trimesh.voxel import creation
from sklearn.model_selection import train_test_split
from mathutils import Vector

cwd = os.getcwd()

OBJ_DIR = '_objects'
IMAGE_DIR = '_images'
VOXEL_RES = 32
IMAGE_RES = 128

OBJ_FILE_FILTER = '*.obj'
if len(sys.argv) >= 5:
    OBJ_FILE_FILTER = sys.argv[4]
print(f'Creating dataset using object file filter: {OBJ_FILE_FILTER}')

def reset_blender_scene(use_homefile=False):
    if (use_homefile):
        bpy.ops.wm.read_homefile(use_empty=True)
    else:
	    bpy.ops.wm.read_factory_settings(use_empty=True)
    
def calculate_object_bounding_box_center(obj):
    local_coords = [Vector(corner) for corner in obj.bound_box]
    om = obj.matrix_world
    world_coords = [om @ coord for coord in local_coords]
    min_corner = Vector(min(world_coords, key=lambda coord: coord[i])[i] for i in range(3))
    max_corner = Vector(max(world_coords, key=lambda coord: coord[i])[i] for i in range(3))
    center = (min_corner + max_corner) / 2
    return center

def render_and_save_image(output_dir, obj_name, x, y, size_percent, angle_x, angle_y):
    image_filename = f'{obj_name}_{x}_{y}_{size_percent}_{angle_x}_{angle_y}.png'
    image_filepath = os.path.join(cwd, output_dir, image_filename)
    bpy.context.scene.render.filepath = image_filepath
    bpy.ops.render.render(write_still=True)
    return image_filepath
    
def get_image_matrix(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_RES, IMAGE_RES))
    img = img / 255.0
    return img
    
def get_obj_matrix(obj, voxel_resolution=VOXEL_RES):
    #obj_mesh = obj.to_mesh()
    #polygons = obj_mesh.polygons
    # Get the vertices from the mesh
    #vertices = [v.co for v in obj_mesh.vertices]
    # Get the faces from the mesh as indices
    #faces = np.array([p.vertices[:] for p in obj_mesh.polygons], dtype=np.int64)
    #faces = [p.vertices for p in obj_mesh.polygons]
    # Convert the 'faces' list into a list of lists
    #faces_list = [list(poly.vertices) for poly in polygons]
    # Convert the list of lists into a 2D numpy array
    #faces = np.array(faces_list, dtype=np.int64)
    #print("Faces:", faces)
    #mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = get_obj_trimesh(obj)
    pitch = mesh.extents.max() / voxel_resolution
    voxels = trimesh.voxel.creation.voxelize(mesh, pitch)
    voxel_matrix = voxels.matrix.astype(np.float32)
    # Clip the dimensions of the voxel matrix to the maximum allowed voxel_resolution
    clipped_voxel_matrix = voxel_matrix[:voxel_resolution, :voxel_resolution, :voxel_resolution]
    # Pad the voxel matrix with zeros to make it a cube of shape (voxel_resolution, voxel_resolution, voxel_resolution)
    padded_voxel_matrix = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
    padded_voxel_matrix[:clipped_voxel_matrix.shape[0], :clipped_voxel_matrix.shape[1], :clipped_voxel_matrix.shape[2]] = clipped_voxel_matrix
    return padded_voxel_matrix
    
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
        
        ## Translate the mesh
        #print(f'obj location: {obj.location}')
        #print(f'mesh location: {mesh.centroid}')
        #mesh.apply_translation([obj.location.x, obj.location.y, obj.location.z])
        #print(f'mesh location moved: {mesh.centroid}')
        
        ## Scale the mesh.
        #print(f"obj scale: {obj.scale}")
        #print(f"mesh scale: {mesh.extents}")
        #mesh.apply_scale(obj.scale[0])
        #print(f"mesh scale updated: {mesh.extents}")
        
        return mesh
    else:
        print("Error: 'obj' is not a mesh object.")
        exit(1)
    
def translate_obj(obj, x_offset, y_offset, z_offset):
    obj.location.x = obj.location.x + x_offset
    obj.location.y = obj.location.y + y_offset
    obj.location.z = obj.location.z + z_offset
    
def set_obj_location(obj, x, y, z):
    obj.location.x = x
    obj.location.y = y
    obj.location.z = z
	
def set_camera_location(camera, center, angle_x, angle_y, distance):
    camera.rotation_euler[0] = math.radians(angle_x)
    camera.rotation_euler[1] = math.radians(angle_y)
    camera.rotation_euler[2] = 0

    camera.location.x = center.x + distance * math.sin(math.radians(angle_x)) * math.cos(math.radians(angle_y))
    camera.location.y = center.y + distance * math.sin(math.radians(angle_x)) * math.sin(math.radians(angle_y))
    camera.location.z = center.z + distance * math.cos(math.radians(angle_x))

    constraint = camera.constraints.get('Track To')
    if constraint is None:
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = bpy.context.selected_objects[0]
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
	
def set_camera_location_center(angle_x, angle_y, distance):
    # Set camera location to center (less accurate for complex polygons).
    camera = bpy.data.objects['Camera']

    # Set camera rotation
    camera.rotation_euler[0] = math.radians(angle_x)
    camera.rotation_euler[1] = math.radians(angle_y)
    camera.rotation_euler[2] = 0

    # Set camera location
    camera.location.x = distance * math.sin(math.radians(angle_x)) * math.cos(math.radians(angle_y))
    camera.location.y = distance * math.sin(math.radians(angle_x)) * math.sin(math.radians(angle_y))
    camera.location.z = distance * math.cos(math.radians(angle_x))
	
	# Add Track To constraint to make the camera point towards the object
    constraint = camera.constraints.get('Track To')
    if constraint is None:
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = bpy.context.selected_objects[0]  # Assumes the 3D model is the only selected object
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
    
def set_obj_rotation(obj, angle_x, angle_y, angle_z=0):
    obj.rotation_euler[0] = math.radians(angle_x)
    obj.rotation_euler[1] = math.radians(angle_y)
    obj.rotation_euler[2] = math.radians(angle_z)
    
def set_obj_size(obj, scale=1.0):
    obj.scale = (scale, scale, scale)

image_data = []
obj_data = []
def populate_dataset(obj_path, output_dir):
    obj_name = os.path.basename(obj_path)
    
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Import 3D model
    bpy.ops.import_scene.obj(filepath=obj_path)

    # Set render resolution and format
    bpy.context.scene.render.resolution_x = IMAGE_RES
    bpy.context.scene.render.resolution_y = IMAGE_RES
    bpy.context.scene.render.image_settings.file_format = 'PNG'
	
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # Enable transparency
    bpy.context.scene.render.film_transparent = True

    # Set output file format to RGBA
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup camera
    obj = bpy.context.selected_objects[0]
    obj_center = calculate_object_bounding_box_center(obj)
    max_dim = max(obj.dimensions)
    distance = max_dim * 2
     
     # Set camera to orthographic projection
    camera = bpy.data.objects['Camera']
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max_dim * 3   # zoom out
    
    # Set up environment light
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    world_nodes.clear()
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs['Strength'].default_value = 1.0  # Adjust the strength of the environment light
    background.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1)  # Set the color to gray
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    # Update camera location
    #set_camera_location(camera, obj_center, 0, 0, distance)
    
    # Get camera properties.
    cam = camera
    cam_matrix = cam.matrix_world
    cam_loc = cam_matrix.to_translation()
    cam_rot = cam_matrix.to_euler()
    cam_scale = cam_matrix.to_scale()
    cam_data = cam.data
    cam_angle_x = cam_data.angle_x
    cam_angle_y = cam_data.angle_y
    half_width = math.tan(cam_angle_x / 2) * cam_loc.z
    half_height = math.tan(cam_angle_y / 2) * cam_loc.z
    x_step_size = 1
    y_step_size = 1
    x_range = int((2 * half_width) / x_step_size)
    y_range = int((2 * half_height) / y_step_size)
    
    # Object translation settings (TODO: Implement translation logic).
    min_location_offset = -1
    max_location_offset = 1
    location_offset_step = 1
    
    # Object rotation settings.
    min_angle_x = 0
    min_angle_y = 0
    min_angle_z = 0
    max_angle_x = 90
    max_angle_y = 90
    max_angle_z = 0
    angle_step=45
    
    # Object scaling settings.
    min_size_percent = 10
    max_size_percent = 20
    size_step = 50
    
    # Save the object's initial location and rotation to help with resetting the object within the loops below.
    obj_init_location = (obj.location.x, obj.location.y, obj.location.z)
    obj_init_rotation = (np.rad2deg(obj.rotation_euler[0]), np.rad2deg(obj.rotation_euler[1]), np.rad2deg(obj.rotation_euler[2]))

    # Capture images from multiple perspectives
    # TODO: Translate / rotate camera to capture images from multiple perspectives (?). Need to determine if there is benefit to doing this. -JMS
    # TODO: Translate object to capture images of object at varying locations (?).  Need to determine if there is benefit to doing this. -JMS
    for x in range(-x_range // 2, x_range // 2, x_step_size):
        for y in range(-y_range // 2, y_range // 2, y_step_size):
            # Translate the object.
            set_obj_location(obj, x, y, 0)
            #set_obj_location(camera, x, y, 0)
            
            for angle_x in range(min_angle_x, max_angle_x, angle_step):
                for angle_y in range(min_angle_y, max_angle_y, angle_step):
                    # Rotate the object.
                    set_obj_rotation(obj, angle_x, angle_y)
                    for size_percent in range(min_size_percent, max_size_percent, size_step):
                        # Resize the object.
                        size_ratio = size_percent / 100.0
                        set_obj_size(obj, size_ratio)
                        # Render and save an image of the object.
                        image_filepath = render_and_save_image(output_dir, obj_name, x, y, size_percent, angle_x, angle_y)
                        # Convert the image and object to matricies.
                        image_matrix = get_image_matrix(image_filepath)
                        obj_matrix = get_obj_matrix(obj)
                        # Append the image and object data to the dataset.
                        image_data.append(image_matrix)
                        obj_data.append(obj_matrix)

# Populate the dataset.
obj_paths = glob.glob(os.path.join(OBJ_DIR, OBJ_FILE_FILTER))
for obj_path in obj_paths:
    populate_dataset(obj_path, IMAGE_DIR)

# Create the dataset output.
images_2d = np.array(image_data)
objs_3d = np.array(obj_data)

# Split the dataset into training and validation sets
train_images, val_images, train_objs, val_objs = train_test_split(images_2d, objs_3d, test_size=0.2, random_state=42)

# Save the datasets as NumPy files for future use
np.save('train_images.npy', train_images)
np.save('val_images.npy', val_images)
np.save('train_objs.npy', train_objs)
np.save('val_objs.npy', val_objs)