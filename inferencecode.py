"""
This script contains functions for making inferences on images using
a pre-trained model for NSFW (Not Safe For Work) content detection.
It integrates with FastAPI to serve inference requests through HTTP endpoints.

Functions:
1. download_file(url: str) -> bytes:
    Downloads a file from the specified URL and returns its content as bytes.

2. decision_function(result, image_path: str) -> JSONResponse:
    Analyzes the classification result and decides whether an image contains NSFW content or not.
    Writes analytics to files and returns a JSON response.

3. make_inference(url: str = Form(None),
                            file: UploadFile = File(None),
                            image_path: str = Form(None)) -> JSONResponse:
    Handles inference requests by downloading the image file from a URL, uploading a file,
    or specifying a local file path. Performs NSFW content detection and returns a JSON response.

"""
from datetime import datetime
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse
import requests
from nsfw_detector import predict
from ml_data_utils import load_image_from_bytes, load_image_from_url, download_file
import traceback

IMAGE_DIM = 299
# Load the model when the application starts
MODEL_PATH = 'inception_v3.h5'
model = predict.load_model(MODEL_PATH)

def log_result(result, image_path, norm_or_banned):
    """
    Logs the result of image classification.

    Args:
        result: The result of the image classification.
        image_path (str): The path of the image.
        norm_or_banned (str): The file to log the result in.
    """
    url = "http://localhost:8000/feedback"
    payload = {"decision": result, "image_path": image_path}
    response = requests.post(url, data=payload)

    if norm_or_banned == 'NORM_IMAGES.txt':
        data = {"decision": False, "detailed_info": result}
        status = False
        if response == 1:
            with open(norm_or_banned, 'a', encoding='utf-8') as f:
                f.write(f"\n{image_path} ---> {result} TIME: {datetime.now()}")
        else:
            with open('BANNED_IMAGES.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{image_path} ---> {result} TIME: {datetime.now()}")
    elif norm_or_banned == 'BANNED_IMAGES.txt':
        data = {"decision": True, "detailed_info": result}
        status = True
        if response == 1:
            with open(norm_or_banned, 'a', encoding='utf-8') as f:
                f.write(f"\n{image_path} ---> {result} TIME: {datetime.now()}")
        else:
            with open('NORM_IMAGES.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{image_path} ---> {result} TIME: {datetime.now()}")
    else:
        data = {"decision": None, "detailed_info": result}
        status = None
    return JSONResponse(content={"status": status,
                                             "message": "СКАЙНЕТ РАБОТАЕТ",
                                             "code": "SS-10000", "data": data})

def decision_function(result, image_path):
    """
    Analyzes the result of image classification and returns the decision.

    Args:
        result: The result of the image classification.
        image_path (str): The path of the image.

    Returns:
        JSONResponse: The JSON response containing the decision.
    """
    try:
        nsfw_threshold = 0.95
        nsfw_categories = ["snail", "slug"]
        max_probs = sorted(result[0].items(), key=lambda x: x[1], reverse=True)[:2]
        for category, prob in max_probs:
            # checking for NSFW image presence
            if category in nsfw_categories and prob > nsfw_threshold:
                # collect analytics for moderation model upgrade
                decision_data = log_result(max_probs, image_path, norm_or_banned)

            # if image category not in the nsfw list
            # collect analytics for moderation model upgrade
            else:
                decision_data = log_result(max_probs, image_path)
            return decision_data

    except Exception as e:
        traceback.print_exc()  # Print the traceback for debugging
        return JSONResponse(content={"Message": str(e)})

def make_inference(file: UploadFile = File(None),
                            image_path: str = Form(None)):
    """
    Makes inference on the provided image data.

    Args:
        url (str, optional): URL of the image. Defaults to None.
        file (UploadFile, optional): Uploaded image file. Defaults to None.
        image_path (str, optional): Path of the image. Defaults to None.

    Returns:
        JSONResponse: The JSON response containing the inference result.

    Raises:
        ValueError: If URL, file, or image_path was not provided.

    """

    try:
        downloaded_image, downloaded_path = download_file(file, image_path)
        nd_image = load_image_from_bytes(downloaded_image)
        result = predict.classify_nd(model, nd_image)
        return decision_function(result, downloaded_path)
    except Exception as e:
        return JSONResponse(content={"status": False,
                                                             "message": "ERROR",
                                                             "code": "SS-10000",
                                                             "data": str(e)})