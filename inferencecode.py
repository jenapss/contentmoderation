from nsfw_detector import predict
# import argparse
# import json
# from os import listdir
# from os.path import isfile, join, exists, isdir, abspath

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_hub as hub

# IMAGE_DIM = 299

# def load_images(image_paths, image_size, verbose=True):
#     '''
#     Function for loading images into numpy arrays for passing to model.predict
#     inputs:
#         image_paths: list of image paths to load
#         image_size: size into which images should be resized
#         verbose: show all of the image path and sizes loaded

#     outputs:
#         loaded_images: loaded images on which keras model can run predictions
#         loaded_image_indexes: paths of images which the function is able to process

#     '''
#     loaded_images = []
#     loaded_image_paths = []

#     if isdir(image_paths):
#         parent = abspath(image_paths)
#         image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
#     elif isfile(image_paths):
#         image_paths = [image_paths]

#     for img_path in image_paths:
#         try:
#             if verbose:
#                 print(img_path, "size:", image_size)
#             image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
#             image = keras.preprocessing.image.img_to_array(image)
#             image /= 255
#             loaded_images.append(image)
#             loaded_image_paths.append(img_path)
#         except Exception as ex:
#             print("Image Load Failure: ", img_path, ex)
    
#     return np.asarray(loaded_images), loaded_image_paths

# def classify_nd(model, nd_images, predict_args={}):
#     """
#     Classify given a model, image array (numpy)
    
#     Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
#     """
#     model_preds = model.predict(nd_images, **predict_args)
#     # preds = np.argsort(model_preds, axis = 1).tolist()
    
#     categories = ['classabc', 'classabc', 'classabc', 'classabc', 'classabc']

#     probs = []
#     for i, single_preds in enumerate(model_preds):
#         single_probs = {}
#         for j, pred in enumerate(single_preds):
#             single_probs[categories[j]] = float(pred)
#         probs.append(single_probs)
#     return probs


# def classify(model, input_paths, image_dim=IMAGE_DIM, predict_args={}):
#     """
#     Classify given a model, input paths (could be single string), and image dimensionality.
    
#     Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
#     """
#     images, image_paths = load_images(input_paths, (image_dim, image_dim))
#     probs = classify_nd(model, images, predict_args)
#     return dict(zip(image_paths, probs))


# def load_model(model_path):
#     if model_path is None or not exists(model_path):
#     	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
#     model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
#     return model


model = predict.load_model('nsfw.299x299.h5')
# Predict single image
print(predict.classify(model, '1784552416.jpg'))