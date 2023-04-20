import os
import numpy as np
import tensorflow as tf
from skimage import io
from skimage.transform import resize

def stack_images(images):
    return np.concatenate(images, axis=-1)
    
def image_to_3d(images_directory, image_prefix, model_path, output_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the input images
    image_filenames = [f for f in os.listdir(images_directory) if f.startswith(image_prefix)]
    obj_images = []
    for image_filename in image_filenames:
        image_path = os.path.join(images_directory, image_filename)
        image = io.imread(image_path, as_gray=True)
        image_resized = resize(image, (128, 128), anti_aliasing=True)
        obj_images.append(image_resized)

    stacked_images = stack_images(obj_images)
    stacked_images = np.expand_dims(stacked_images, axis=-1)  # Add channel dimension
    stacked_images = np.expand_dims(stacked_images, axis=0)  # Add batch dimension

    # Predict the 3D model using the trained model
    predicted_3d = model.predict(stacked_images)
    predicted_3d = predicted_3d.squeeze()

    # Convert the 3D model to vertices
    vertices = []
    for z in range(predicted_3d.shape[0]):
        for y in range(predicted_3d.shape[1]):
            for x in range(predicted_3d.shape[2]):
                if predicted_3d[z, y, x] > 0.5:  # Threshold for selecting vertices
                    vertices.append((x, y, z))

    # Save the 3D model as an .obj file
    save_obj(vertices, output_path)

# Usage example:
images_directory = '_output'
image_prefix = 'cube.obj'
model_path = 'model.h5'
output_path = '_predict'

image_to_3d(images_directory, image_prefix, model_path, output_path)
