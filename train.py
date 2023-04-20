import cv2
import numpy as np
import tensorflow as tf
import trimesh
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 1)):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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
    model.add(layers.Dense(35937, activation='relu'))  # Updated output size
    model.add(layers.Reshape((33, 33, 33)))  # Updated reshape dimensions

    return model

def stack_images(images):
    return np.concatenate(images, axis=-1)


# Load the dataset
train_images = np.load('train_images.npy')
val_images = np.load('val_images.npy')
train_models = np.load('train_models.npy')
val_models = np.load('val_models.npy')

# Preprocessing
train_images_grouped = []
train_models_grouped = []

for obj_images, obj_model in zip(train_images, train_models):
    stacked_images = stack_images(obj_images)
    stacked_images = np.expand_dims(stacked_images, axis=-1)  # Add channel dimension
    train_images_grouped.append(stacked_images)
    train_models_grouped.append(obj_model)

val_images_grouped = []
val_models_grouped = []

for obj_images, obj_model in zip(val_images, val_models):
    stacked_images = stack_images(obj_images)
    stacked_images = np.expand_dims(stacked_images, axis=-1)  # Add channel dimension
    val_images_grouped.append(stacked_images)
    val_models_grouped.append(obj_model)

train_images_stacked = np.array(train_images_grouped)
train_models_stacked = np.array(train_models_grouped)
val_images_stacked = np.array(val_images_grouped)
val_models_stacked = np.array(val_models_grouped)


# Training
model = create_model(input_shape=train_images_stacked.shape[1:])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

print("train_images_stacked shape:", train_images_stacked.shape)
print("train_models_stacked shape:", train_models_stacked.shape)
print("val_images_stacked shape:", val_images_stacked.shape)
print("val_models_stacked shape:", val_models_stacked.shape)

history = model.fit(train_images_stacked, train_models_stacked, epochs=10, batch_size=32, validation_data=(val_images_stacked, val_models_stacked))

model.save('2d_to_3d_model.h5')
