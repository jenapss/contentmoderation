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
from inferencecode import make_inference, log_result
from ml_data_utils import norm_banned_image_fetch
app = FastAPI()

@app.post("/classify")
async def classify_image(image_path: str = Form(None),
                                        file: UploadFile = File(None)):
    """
    Classify images provided via URL, file upload, or file path.

    Args:
        url (str, optional): URL of the image to classify. Defaults    to None.
        file (UploadFile, optional): Uploaded image file. Defaults to None.
        image_path (str, optional): Local path of the image file. Defaults to None.

    Returns:
        JSONResponse: JSON response containing the classification results.
    """
    inference = make_inference(file = file, image_path = image_path)
   # feedback = get_feedback(inference)
    return inference

@app.post("/feedback")

async def get_feedback(decision, image_path):
    print("")
    print("Please Provide Feedback for the Image and according clasification:")
    print(f"DECISION: {decision}")
    print(f"IMAGE PATH: {image_path}")
    print("0: Classification is INCORRECT")
    print("1: Classification is CORRECT")
    feedback = int(input("Enter Feedback: "))
    if feedback not in [0, 1]:
        print("Invalid Feedback. Please enter 0 or 1.")
        feedback = int(input("Enter Feedback: "))
    return feedback

@app.post("/fetchimages")

async def fetch_images(request_data: dict):
    normal_images_path = request_data.get("normal_images_path", None)
    banned_images_path = request_data.get("banned_images_path", None)
    batch_size = request_data.get("batch_size", 5)

    # if not normal_images_path or not banned_images_path:
    #     return {"error": "Missing required parameters"}

    return norm_banned_image_fetch(normal_images_path, banned_images_path, batch_size)

@app.get("/health")
async def health_check():
    """
    Check the health status of the application.

    Returns:
        JSONResponse: JSON response indicating the health status.
    """
    print("HERE")
    return {
            "status": True,
            "message": "Good Job !!!",
            "code": "SS-10000",
            "data": "СКАЙНЕТ РАБОТАЕТ"
            }
