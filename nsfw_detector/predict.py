"""
This script provides functions for loading images, loading
a pre-trained NSFW (Not Safe For Work) detection model,
classifying images using the loaded model, and decoding the predictions.

Functions:
1. load_images(image_paths, image_size, verbose=True) -> Tuple[np.ndarray, List[str]]:
    Loads images into numpy arrays for passing to model.predict.

2. load_model(model_path: str) -> keras.Model:
    Loads a pre-trained model for NSFW content detection from the specified path.

3. classify(model, input_paths, image_dim=IMAGE_DIM, predict_args={}) -> Dict[str, Dict[str, float]]:
    Classifies images given a model, input paths, and image dimensionality.

4. classify_nd(model, nd_images, predict_args={}) -> List[Dict[str, float]]:
    Classifies images given a model and image array (numpy).

Variables:
- IMAGE_DIM (int): Required/default image dimensionality.

"""
from os import listdir
from os.path import isfile, join, exists, isdir, abspath
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.applications.inception_v3 import decode_predictions

IMAGE_DIM = 299   # required/default image dimensionality

def load_images(image_paths, image_size, verbose=True):
    """
    Loads a pre-trained model for NSFW (Not Safe For Work) content
    detection from the specified path.

    Args:
        model_path (str): The path to the pre-trained model file.

    Returns:
        keras.Model: The loaded pre-trained model.

    Raises:
        FileNotFoundError: If the specified model file is not found.
        ValueError: If the specified model file is invalid or cannot be loaded.
    """
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]

    for img_path in image_paths:
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)

    return np.asarray(loaded_images), loaded_image_paths

def load_model(model_path):
    """
    Loads a pre-trained model for NSFW (Not Safe For Work) content
    detection from the specified path.

    Args:
        model_path (str): The path to the pre-trained model file.

    Returns:
        keras.Model: The loaded pre-trained model.

    Raises:
        FileNotFoundError: If the specified model file is not found.
        ValueError: If the specified model file is invalid or cannot be loaded.
    """
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'KerasLayer': hub.KerasLayer},
                                       compile=False)
    return model



def classify(model, input_paths, image_dim=IMAGE_DIM, predict_args={}):
    """
    Classify given a model, input paths (could be single string), and image dimensionality.

    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd(model, images, predict_args)
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images, predict_args={}):
    """
    Classify given a model, image array (numpy)

    Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
    """
    model_preds = model.predict(nd_images, **predict_args)
    # the model returns 1000 outputs
    # decode the prediction output to get only
    decoded_preds = decode_predictions(model_preds, top=5)
    categories = ['snail', 'slug', 'tiger', 'tiger_cat', 'leopard']
    probs = []
    for single_preds in decoded_preds:
        single_probs = {}
        for label, description, probability in single_preds:
            # Map the description to your predefined categories
            if description in categories:
                single_probs[description] = float(probability)
        for label, description, probability in single_preds:
            # Map the description to your predefined categories
            if description in categories:
                single_probs[description] = float(probability)
        probs.append(single_probs)
    return probs
