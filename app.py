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
app = FastAPI()

# Load the model when the application starts
model_path = 'inception_v3.h5'
model = predict.load_model(model_path)

def download_file(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def load_image_from_bytes(image_content: bytes, image_size=(299, 299)):
    #print("00000000",type(image_content))
    try:
        image = keras.preprocessing.image.load_img(io.BytesIO(image_content), target_size=image_size)
    except PIL.UnidentifiedImageError:
        # Load the HEIC image from the bytearray
        heif_file = pillow_heif.from_bytes(data=image_content,size=(299,299), mode="RGB")
        # Convert to a PIL Image
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
                # Convert the PIL Image to a JPG bytearray
        jpg_bytearray = io.BytesIO()
        image.save(jpg_bytearray, format="JPEG")
        jpg_bytearray = jpg_bytearray.getvalue()

        #finally load to keras
        image = keras.preprocessing.image.load_img(io.BytesIO(jpg_bytearray), target_size=image_size)
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize the image
    return image_array

def decision_function(result, image_path):
    try:
        nsfw_threshold = 0.95
        nsfw_categories = ["snail", "slug"]
        max_probs = sorted(result[0].items(), key = lambda x:  x[1], reverse = True)[:2]
        for category, prob in max_probs:
            # checking for NSFW image presense
            if category in nsfw_categories and prob > nsfw_threshold:
                # collect analytics for moderation model upgrade
                with open('BANNED_IMAGES.txt', 'a') as f:
                    f.write("\n{} ---> {} TIME: {}".format(image_path, max_probs, str(datetime.now())))
                data = {"decision": True, "detailed_info": max_probs}
                return JSONResponse(content={ "status": True, "message": "СКАЙНЕТ РАБОТАЕТ", "code": "SS-10000","data": data})

            else:
                # if image category not in the nsfw list
                # collect analytics for moderation model upgrade
                with open('NORM_IMAGES.txt', 'a') as f:
                        f.write("\n{} ---> {} TIME: {}".format(image_path, max_probs,str(datetime.now())))
                        data = {"decision": False, "detailed_info": max_probs}
                        return JSONResponse(content={ "status": False, "message": "СКАЙНЕТ РАБОТАЕТ", "code": "SS-10000","data": data})
    except Exception as e:
        return {"Message": "{}".format(e)}


@app.post("/classify")
async def classify_image(url: str = Form(None), file: UploadFile = File(None), image_path:str = Form(None)):
    # Perform classification using the loaded model
    # result = predict.classify(model, file.filename)

    if file is not None:
        try:
            uploaded_image = file.file.read()
            #print(type(uploaded_image))
            nd_images = load_image_from_bytes(uploaded_image)
            #print(type(nd_images))
            result = predict.classify_nd(model,nd_images)
        except Exception as e:
            #return JSONResponse(content={f"Message": "{}".format(e)})
            return JSONResponse(content={ "status": False, "message": "ERROR", "code": "SS-10000","data": e})

        return decision_function(result, "The image file")

    elif url is not None:
        try:
            image_content = download_file(url)
        except Exception as e:
            #return JSONResponse(content={f"Message": "{}".format(e)})
            return JSONResponse(content={ "status": False, "message": "ERROR", "code": "SS-10000","data": e})
        #print(type(image_content))
        nd_images = load_image_from_bytes(image_content)
        result = predict.classify_nd(model, nd_images)
        return decision_function(result, url)

    elif image_path is not None:
        # read image file
        print("IMAGE PATH ---->",image_path)
        try:
            img_bytes_io = io.BytesIO(open(image_path, 'rb').read())
        except Exception as e:
            #return JSONResponse(content={f"Message": "{}".format(e)})
            return JSONResponse(content={ "status": False, "message": "ERROR", "code": "SS-10000","data": e})

        nd_image = load_image_from_bytes(img_bytes_io.read())
        result = predict.classify_nd(model, nd_image)
        return decision_function(result,image_path)
    #return JSONResponse(content={"Message": "No file or URL provided"})
    return JSONResponse(content={ "status": False, "message": "ERROR", "code": "SS-10000","data": "URL | FILE | IMAGE_PATH was not provided"})


@app.get("/health")
async def health_check():
    print("HERE")
    return {
            "status": True,
            "message": "Molodec !!!",
            "code": "SS-10000",
            "data": "СКАЙНЕТ РАБОТАЕТ"
            }
