import os
import io
import traceback
import yaml
from datetime import datetime
from fastapi.responses import JSONResponse
from nsfw_detector import predict
from ml_data_utils import load_images_from_bytes, download_file

# Load constants
def load_constants(file_path):
    with open(file_path, 'r') as file:
        constants = yaml.safe_load(file)
    return constants

constants = load_constants('constants.yaml')
returning = constants["RETURN_MESSAGES"]
IMAGE_DIM = constants['IMAGE_DIM']
MODEL_PATH = constants['MODEL_PATH']

# Load the model when the application starts
model = predict.load_model(MODEL_PATH)

def decision_function(result, image_path):
    """
    Analyzes the result of image classification and returns the decision.

    Args:
        result: The result of the image classification.
        image_path (str): The path of the image.

    Returns:
        dict: The dictionary containing the decision.
    """
    try:
        nsfw_threshold = 0.95
        nsfw_categories = ["snail", "slug"]
        max_probs = sorted(result[0].items(), key=lambda x: x[1], reverse=True)[:2]
        for category, prob in max_probs:
            # checking for NSFW image presence
            if category in nsfw_categories and prob > nsfw_threshold:
                data = {"decision": True, "detailed_info": max_probs}
            else:
                data = {"decision": False, "detailed_info": max_probs}

            return {
                "status": data["decision"],
                "message": "СКАЙНЕТ РАБОТАЕТ",
                "code": "SS-10000",
                "data": data,
                "image_path": image_path
            }
    except Exception as e:
        traceback.print_exc()  # Print the traceback for debugging
        return {"Message": str(e)}

def make_inference(image_path: str, image_file: bytes):
    """
    Makes inference on the provided image data.

    Args:
        image_data (bytes): Image data in bytes.
        image_path (str): Optional parameter to specify the image path.

    Returns:
        dict: The dictionary containing the inference result.
    """
    try:
        # Load image and make inference
        nd_image = load_images_from_bytes(image_file)
        result = predict.classify_nd(model, nd_image)
        return decision_function(result, image_path)
    except Exception as e:
        global returning
        returning["status"] = False
        returning["message"] = "ERROR"
        returning["code"] = "SS-10000"
        returning["data"] = str(e)
        return returning
