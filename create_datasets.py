import glob
import os
import subprocess
import sys

OBJ_DIR = '_objects'
OBJ_FILE_FILTER = '*.obj'
if len(sys.argv) >= 2:
    OBJ_FILE_FILTER = sys.argv[1]

print(f'Creating datasets using object file filter: {OBJ_FILE_FILTER}')

obj_paths = glob.glob(os.path.join(OBJ_DIR, f"{OBJ_FILE_FILTER}"))

for obj_path in obj_paths:
    obj_file = os.path.basename(obj_path)
    subprocess.run(["blender", "-b", "-P", "create_dataset.py", obj_file])
    