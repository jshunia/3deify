import bpy
import os
import math
from mathutils import Vector

cwd = os.getcwd()

def calculate_object_bounding_box_center(obj):
    local_coords = [Vector(corner) for corner in obj.bound_box]
    om = obj.matrix_world
    world_coords = [om @ coord for coord in local_coords]
    min_corner = Vector(min(world_coords, key=lambda coord: coord[i])[i] for i in range(3))
    max_corner = Vector(max(world_coords, key=lambda coord: coord[i])[i] for i in range(3))
    center = (min_corner + max_corner) / 2
    return center
	
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

def render_and_save(output_dir, model_name, angle_x, angle_y):
    output_file = os.path.join(cwd, output_dir, f'{model_name}_{angle_x}_{angle_y}.png')
    bpy.context.scene.render.filepath = output_file
    bpy.ops.render.render(write_still=True)

def create_dataset(model_path, output_dir, distance=0, angle_step=45):
    model_name = os.path.basename(model_path)
    
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Import 3D model
    bpy.ops.import_scene.obj(filepath=model_path)

    # Set render resolution and format
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
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
    camera.data.ortho_scale = max_dim * 1.5
    
    #distance = max_dim * 2
    #bpy.data.cameras['Camera'].lens = max_dim * 1.5
    #camera = bpy.data.objects['Camera']
    
    # Set up environment light
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    world_nodes.clear()
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs['Strength'].default_value = 1.0  # Adjust the strength of the environment light
    #background.inputs['Color'].default_value = (1, 1, 1, 1)  # Set the color to white
    background.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1)  # Set the color to gray
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    # Capture images from multiple perspectives
    for angle_x in range(0, 360, angle_step):
        for angle_y in range(0, 360, angle_step):
            #set_camera_location_center(angle_x, angle_y, distance) # Set camera location to center (less accurate for complex polygons).
            set_camera_location(camera, obj_center, angle_x, angle_y, distance)
            camera.data.ortho_scale = max_dim * (1.5 + abs(math.sin(math.radians(angle_x))))  # Adjust ortho_scale dynamically
            render_and_save(output_dir, model_name, angle_x, angle_y)

# Example usage
# To run the script, save it to a file (e.g., create_dataset.py) and run it using Blender's Python: 
# blender -b -P create_dataset.py

model_file = 'cube.obj'
model_dir = '_input'
output_dir = '_output'
model_path = f'{model_dir}/{model_file}'
create_dataset(model_path, output_dir)
