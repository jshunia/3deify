import numpy as np
import tensorflow as tf
from skimage import io
from skimage.transform import resize

def save_obj(vertices, filename):
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

def image_to_3d(image_path, model_path, output_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the input image
    image = io.imread(image_path, as_gray=True)
    image_resized = resize(image, (128, 128), anti_aliasing=True)
    image_resized = np.expand_dims(image_resized, axis=-1)  # Add channel dimension
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Predict the 3D model using the trained model
    predicted_3d = model.predict(image_resized)
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
image_path = '_output/cube.obj_45_45.png'
model_path = '2d_to_3d_model.h5'
output_path = '_predict'

image_to_3d(image_path, model_path, output_path)
