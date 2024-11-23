"""
File: app.py

Description:
This file contains a FastAPI application for image classification and health check endpoints.

Endpoints:
1. POST /classify:
   - Description: Endpoint for classifying images provided via URL, file upload, or file path.
   - Parameters:
     - url (str, optional): URL of the image to classify.
     - file (UploadFile, optional): Uploaded image file.
     - image_path (str, optional): Local path of the image file.
   - Returns:
     - JSONResponse: JSON response containing the classification results.

2. GET /health:
   - Description: Endpoint for health check.
   - Returns:
     - JSONResponse: JSON response indicating the health status of the application.

Example Usage:
1. To classify an image:
   POST request to http://localhost:8000/classify with form
   data containing: 'url', 'file', or 'image_path'.

2. To perform a health check:
   GET request to http://localhost:8000/health.
"""
from fastapi import FastAPI, File, Form, UploadFile
from inferencecode import make_inference
from ml_data_utils import norm_banned_image_fetch, log_results,  load_constants
from fastapi.responses import JSONResponse
import yaml

app = FastAPI()
constants = load_constants('constants.yaml')
misclassification = constants['MISCLASSIFICATION']
returning = constants["RETURN_MESSAGES"]
mini_counter = 0



def update_misclassification_count(count):
    with open('constants.yaml', 'r', encoding='utf8') as yaml_file:
        config = yaml.safe_load(yaml_file)

    config['MISCLASSIFICATION'] = count

    with open('constants.yaml', 'w') as yaml_file:
        yaml.dump(config, yaml_file)

@app.post("/classify")
async def classify_image(image_path: str = Form(None),
                         file: UploadFile = File(None)):
    """
    Classify images provided via URL or file upload

    Args:
        image_path (str, optional): URL or image file path of the image to classify. Defaults to None.
        file (UploadFile, optional): Uploaded image file. Defaults to None.

    Returns:
        JSONResponse: JSON response containing the classification results.
    """

    return make_inference(file=file, image_path=image_path)



@app.post("/feedback")
async def request_feedback(image_path: str, classification_decision: dict, verdict: int):
    """
    Receive feedback on the classification decision.
    Works with such command line prompt:
    curl -X POST -H "Content-Type: application/json" -d "{
        "status": true,
        "data": "{
                "detailed_info":
                    [["leopard", 0.9076730608940125]]}
            }"
            "http://localhost:8000/feedback?image_path=/path/to/image.jpg&verdict=1"
    """
    log_results(image_path, classification_decision, verdict)
    global misclassification
    global mini_counter
    global returning
    if verdict == 0:
        misclassification+=1
        mini_counter+=1
        if mini_counter == 10:
            mini_counter = 0
            update_misclassification_count(misclassification)
    if misclassification >= 10:
        trigger = await train_trigger()
        returning["status"] = True
        returning["message"] = trigger
        return returning

    returning["status"] = True
    returning["message"] = "Feedback Received!"
    return returning

async def fetch_images(request_data: dict):
    normal_images_path = request_data.get("normal_images_path", None)
    banned_images_path = request_data.get("banned_images_path", None)
    batch_size = request_data.get("batch_size", 5)
    return norm_banned_image_fetch(normal_images_path, banned_images_path, batch_size)

async def train_trigger():

    return "Training started"

@app.get("/health")
async def health_check():
    """
    Check the health status of the application.

    Returns:
        JSONResponse: JSON response indicating the health status.
    """
    print("HERE")
    global returning
    returning["status"] = True
    returning["message"] = "Well Done !!!"
    return returning