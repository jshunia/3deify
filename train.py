# 3deify - Copyright (C) Joseph M. Shunia, 2023
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# Constants
NUM_DIMS = 3  # Define the number of dimensions for each point in the point cloud
VOXEL_RESOLUTION = 32

# Functions from previous responses
def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
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
    model.add(layers.Dense(VOXEL_RESOLUTION*VOXEL_RESOLUTION*VOXEL_RESOLUTION, activation='relu')) # Change the output size to x*y*z
    model.add(layers.Reshape((VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION))) # Change the output shape to (x, y, z)

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
train_models = np.load('train_models.npy')
val_models = np.load('val_models.npy')

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Use the first GPU device
    gpu_device = gpus[0].name
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    print(f'Using GPU device: {gpu_device}')
else:
    print('GPU not found, using CPU.')

# Create the model
model = create_model()

# Define the model file name and check if it exists
model_file = 'model.h5'
if os.path.exists(model_file):
    # Load the model from the file
    model = tf.keras.models.load_model(model_file)
    print(f'Loaded existing model: {model_file}')
else:
	 print(f'Creating new model: {model_file}')

# Compile the model
# TODO: Figure out which optimizer works best, and what settings are best. Nadam and Adam optimizers both seem to work well.
model.compile(optimizer=optimizers.Nadam(learning_rate=0.001),
              #loss=chamfer_distance, metrics=[chamfer_distance])
			  loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(train_images, train_models, epochs=5, batch_size=32, validation_data=(val_images, val_models))

# Save the model
model.save(model_file)
