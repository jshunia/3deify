import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

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
    model.add(layers.Dense(35937, activation='relu')) # Change the output size to 33*33*33 = 35937
    model.add(layers.Reshape((33, 33, 33))) # Change the output shape to (33, 33, 33)

    return model

# Load your dataset
train_images = np.load('train_images.npy')
val_images = np.load('val_images.npy')
train_models = np.load('train_models.npy')
val_models = np.load('val_models.npy')

# Create the model
model = create_model()

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Train the model
history = model.fit(train_images, train_models, epochs=100, batch_size=32, validation_data=(val_images, val_models))

# Save the model
model.save('model.h5')
