# 3deify - Copyright (C) Joseph M. Shunia, 2023
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# Constants
ENABLE_GPUS = False
MODEL_FILE = 'model.h5'
NUM_DIMS = 3  # Define the number of dimensions for each point in the point cloud
VOXEL_RES= 32
IMAGE_RES= 128

EPOCHS = 10
if len(sys.argv) >= 2:
    EPOCHS = int(sys.argv[1])

print(f'Training model for {EPOCHS} epoch(s)')

# Functions from previous responses
def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_RES, IMAGE_RES, NUM_DIMS)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(VOXEL_RES*VOXEL_RES*VOXEL_RES, activation='relu')) # Change the output size to x*y*z
    model.add(layers.Reshape((VOXEL_RES, VOXEL_RES, VOXEL_RES))) # Change the output shape to (x, y, z)

    return model
	
def pairwise_distance(point_cloud):
	num_points = point_cloud.shape[0]
	expanded = tf.expand_dims(point_cloud, 1)
	expanded = tf.reshape(expanded, [num_points, -1, NUM_DIMS])
	tiled = tf.tile(expanded, [1, num_points, 1])
	transposed = tf.transpose(point_cloud, [1, 0])
	squared_diff = tf.reduce_sum(tf.square(tiled - transposed), axis=-1)
	return tf.reduce_min(squared_diff, axis=1)

def chamfer_distance(point_clouds, predicted):
    dists = tf.map_fn(lambda x: pairwise_distance(x[0], x[1]), (point_clouds, predicted), dtype=tf.float32)
    loss = tf.reduce_mean(tf.reduce_min(dists, axis=1)) + tf.reduce_mean(tf.reduce_min(dists, axis=2))
    return loss

# Load your dataset
train_images = np.load('train_images.npy')
val_images = np.load('val_images.npy')
train_objs = np.load('train_objs.npy')
val_objs = np.load('val_objs.npy')

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
#print(f'gpus:\n{gpus}\n')

if ENABLE_GPUS:
    print("Num GPUs Available: ", len(gpus))
    if gpus:
        # Use the first GPU device
        gpu_device = gpus[0].name
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.set_memory_growth(gpus[0], True)
        print(f'Using GPU device: {gpu_device}')
    else:
        print('GPU not found, using CPU.')

if EPOCHS <= 0:
    print('Training epochs <= 0. Exiting.')
    exit(0)

# Create the model
model = create_model()

# Define the model file name and check if it exists

if os.path.exists(MODEL_FILE):
    # Load the model from the file
    model = tf.keras.models.load_model(MODEL_FILE)
    print(f'Loaded existing model: {MODEL_FILE}')
else:
	 print(f'Creating new model: {MODEL_FILE}')

# Compile the model
# TODO: Figure out which optimizer works best, and what settings are best. Nadam and Adam optimizers both seem to work well.
model.compile(optimizer=optimizers.Nadam(learning_rate=0.001),
              #loss=chamfer_distance, metrics=[chamfer_distance]) # TODO: Figure out how to get Chamfer Distance to work properly and determine if it is a better loss function for training this model.
			  loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(train_images, train_objs, epochs=EPOCHS, batch_size=64, validation_data=(val_images, val_objs))

# Save the model
model.save(MODEL_FILE)
