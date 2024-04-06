    # url = "http://localhost:8000/feedback"
    # payload = {"image_path": image_path,
    #                  "classification_decision": result}
    # response = requests.post(url, json=payload)
    # feedback = json.loads(response.content.decode('utf-8'))
    # operator_feedback = feedback["feedback"]
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
import json
import requests
from nsfw_detector import predict
from ml_data_utils import load_image_from_bytes, load_image_from_url, download_file
import traceback
import requests
import yaml

def load_constants(file_path):
    with open(file_path, 'r') as file:
        constants = yaml.safe_load(file)
    return constants
constants = load_constants('constants.yaml')

#NUMBER_OF_MISCLASSIFICATIONS = 0
IMAGE_DIM = constants['IMAGE_DIM']
# Load the model when the application starts
MODEL_PATH = constants['MODEL_PATH']
model = predict.load_model(MODEL_PATH)

def log_results(image_path, classification_decision, feedback):
    """
    Logs the feedback for the image classification.

    Args:
        image_path (str): The path of the image.
        classification_decision (str): The classification decision (NSFW or SFW).
        feedback (int): The feedback provided by the operator (0 or 1).

    Returns:
        None
    """
    data = json.loads(classification_decision)
    decision = data["data"]["decision"]
    if decision == True:
        if feedback == 1:
            filename = 'BANNED_IMAGES.txt'
        else:
            filename = 'NORM_IMAGES.txt'
            #NUMBER_OF_MISCLASSIFICATIONS += 1
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n{image_path} ---> {max_probs} TIME: {datetime.now()}")
    else:
        if feedback == 1:
            filename = 'NORM_IMAGES.txt'
        else:
            filename = 'BANNED_IMAGES.txt'
            #NUMBER_OF_MISCLASSIFICATIONS += 1
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{image_path} ---> {max_probs} TIME: {datetime.now()}")

def decision_function(result, image_path):
    """
    Analyzes the result of image classification and returns the decision.

    Args:
        result: The result of the image classification.
        image_path (str): The path of the image.

    Returns:
        JSONResponse: The JSON response containing the decision.
    """
    #global NUMBER_OF_MISCLASSIFICATIONS
    try:
        response = {"image_path":"/path/to/image.jpg","classification_decision":"NSFW","feedback": 0}
        feedback = response['feedback']
        nsfw_threshold = 0.95
        nsfw_categories = ["snail", "slug"]
        max_probs = sorted(result[0].items(), key=lambda x: x[1], reverse=True)[:2]
        for category, prob in max_probs:
            # checking for NSFW image presence
            if category in nsfw_categories and prob > nsfw_threshold:
                data = {"decision": True, "detailed_info": max_probs}
            else:
                data = {"decision": False, "detailed_info": max_probs}

            return JSONResponse(content={"status": data["decision"],
                                        "message": "СКАЙНЕТ РАБОТАЕТ",
                                        "code": "SS-10000", "data": data, })

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
#СКАЙНЕТ РАБОТАЕТ