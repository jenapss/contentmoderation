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
app = FastAPI()

@app.post("/classify")
async def classify_image(url: str = Form(None),
                                        file: UploadFile = File(None),
                                        image_path: str = Form(None)):
    """
    Classify images provided via URL, file upload, or file path.

    Args:
        url (str, optional): URL of the image to classify. Defaults to None.
        file (UploadFile, optional): Uploaded image file. Defaults to None.
        image_path (str, optional): Local path of the image file. Defaults to None.

    Returns:
        JSONResponse: JSON response containing the classification results.
    """

    return make_inference(url, file, image_path)

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
            "message": "Molodec !!!",
            "code": "SS-10000",
            "data": "СКАЙНЕТ РАБОТАЕТ"
            }
