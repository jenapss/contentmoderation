from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from nsfw_detector import predict
import pillow_heif
import requests
import keras
import numpy as np
import io
import PIL
from datetime import datetime
from PIL import Image
from inferencecode.py import make_inference
app = FastAPI()

@app.post("/classify")
async def classify_image(url: str = Form(None), file: UploadFile = File(None), image_path: str = Form(None)):

        return make_inference(url, file, image_path)

@app.get("/health")
async def health_check():
    print("HERE")
    return {
            "status": True,
            "message": "Molodec !!!",
            "code": "SS-10000",
            "data": "СКАЙНЕТ РАБОТАЕТ"
            }
