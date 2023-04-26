import glob
import os
import subprocess
import sys

OBJ_DIR = '_objects'
OBJ_FILE_FILTER = '*.obj'
if len(sys.argv) >= 2:
    OBJ_FILE_FILTER = sys.argv[1]

# Create image dataset
create_datasets_result = subprocess.run(['python', 'create_datasets.py', OBJ_FILE_FILTER])
if create_datasets_result.returncode != 0:
        print("create_datasets.py failed, exiting.")
        exit(1)
        
# Preprocess dataset for training
preprocess_result = subprocess.run(['python', 'preprocess.py', OBJ_FILE_FILTER])
if preprocess_result.returncode != 0:
        print("preprocess.py failed, exiting.")
        exit(1)

# Train model
train_result = subprocess.run(['python', 'train.py', '10'])
if train_result.returncode != 0:
        print("train.py failed, exiting.")
        exit(1)
        
print(f'init.py completed successfully.')