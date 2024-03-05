"""
This module provides functions for image preprocessing, parsing, and data validation.

Functions:
1. load_image_from_bytes(image_content: bytes, image_size=(299, 299)) -> np.array:
    Load an image from bytes, preprocess it, and return it as a numpy array.

2. parse_and_download_images(collected_images_path: str) -> None:
    Parse and download images from BANNED & NORM IMAGES text files.

3. check_image_count(filename: str) -> int:
    Count the number of lines in a file.

4. data_validation(train_set_path: str, test_set_path: str) -> bool:
    Check if all images in the training and testing datasets are in valid formats.

Variables:
- DATASET_DEST_PATH: The destination path for the image dataset.
"""
import imghdr
import base64
import os
import io
import PIL
import keras
from PIL import Image
import pillow_heif
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO
from fastapi import File, Form, UploadFile
import logging
from pathlib import Path
import tempfile
DATASET_DEST_PATH = "./img_dataset"
SAVE_PATH = ""
import logging
import requests

logger = logging.getLogger(__name__)

UPLOAD_DIR = "C:\\Users\\sulta\\Downloads\\contentmoderation\\UPLOADED_IMAGES"

def load_image_from_url(url):
    """
    Fetches image data from a given URL.

    Parameters:
    - url (str): The URL of the image to fetch.

    Returns:
    - bytes or None: The binary image data if fetched successfully, or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.exception(f"Error fetching image from URL in load_image_from_url: {e}")
        return None

def save_uploaded_file(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to the specified directory.
    Returns the path where the file is saved.
    """
    # Create the directory if it doesn't exist
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Construct the path to save the file
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)

    try:
        # Ensure the file pointer is at the beginning
        upload_file.file.seek(0)

        # Write the file contents to the specified path
        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())
    except Exception as e:
        # Handle any errors that occur during file saving
        print(f"Error saving file: {e}")
        return None

    return file_path


def download_file(file: UploadFile = File(None),
                            image_path: str = Form(None)):
    """
    Downloads a file from the specified URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
            Tuple[bytes, str]: The content of the downloaded file as bytes and the downloaded path.

    Raises:
        HTTPError: If the HTTP request to the URL fails.
    """
    logger = logging.getLogger(__name__)
    downloaded_path = ""
    if file is not None:
        uploaded_image = file.file.read()
        downloaded_path = save_uploaded_file(file)
    elif image_path is not None:
        logger.info("IMAGE PATH: %s", image_path)  # Add logging here
        with open(image_path, 'rb') as f:
            uploaded_image = f.read()
            downloaded_path = image_path
    else:
        raise ValueError("File or image_path was not provided")

    return uploaded_image, downloaded_path


def load_image_from_bytes(image_content: bytes, image_size=(299, 299)):
    """"
    Turn the images to the correct format -> keras format

    Args:
        image_content : (bytes) the image itself
        image_size: size of image array

    Return:
        Keras preprocessed image array

    """
    try:
        image = keras.preprocessing.image.load_img(io.BytesIO(image_content),
                                                   target_size = image_size)
    except PIL.UnidentifiedImageError:
        # Load the HEIC image from the bytearray
        heif_file = pillow_heif.from_bytes(data=image_content, size=(299,299), mode="RGB")
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
        image = keras.preprocessing.image.load_img(io.BytesIO(jpg_bytearray),
                                                   target_size = image_size)
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize the image
    return image_array

def parse_and_download_images(collected_images_path):
    """
    Parse & download images from BANNED & NORM IMAGES txt files

    args:
    """
    txt_file = open(collected_images_path, 'r')
    lines = txt_file.read().splitlines()
    for line in lines:
        image_path = line.strip().split(" ---> ")
        print(image_path)
    return None

def check_image_count(filename):
    """
    Count number of rows
    """
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

def data_validation(train_set_path, test_set_path):
    """
    Check if all images are inn valid format

    args:
        train_set_path: (str)
        test_set_path: (str)

    returns:
        Boolean - True if all images are valid, False otherwise
    """
    for folder_path in [train_set_path, test_set_path]:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_extension = filename.split('.')[-1].lower()
                image_format = imghdr.what(file_path)
    return image_format

def extract_image_paths(lines):
    """
    Extracts image file paths from lines containing image paths and additional information.

    Parameters:
    - lines (list of str): Lines containing image paths and additional information.

    Returns:
    - list of str: Image file paths extracted from the lines.
    """
    image_paths = []
    for line in lines:
        # Split the line by ' ---> ' to separate the image URL from other information
        parts = line.split(' ---> ')
        if len(parts) > 0:
            # Extract the image URL (first part)
            image_url = parts[0].strip()
            image_paths.append(image_url)
    return image_paths


def local_image_fetch(image_source):
    """
    Fetches the image data from a local source (path, URL, or uploaded file).

    Parameters:
    - image_source (str or file-like object): The source of the image, which can be a local path, URL, or uploaded file.

    Returns:
    - bytes or None: The binary image data if fetched successfully, or None if an error occurs.
    """
    if isinstance(image_source, str):  # If image source is a string (path or URL)
        if os.path.exists(image_source):  # If it's a local path
            try:
                with open(image_source, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"Error fetching image from path: {e}")
                return None
    elif hasattr(image_source, 'read'):  # If it's a file-like object
        try:
            return image_source.read()
        except Exception as e:
            print(f"Error fetching image from uploaded file: {e}")
            return None
    else:
        print("Invalid image source")
        return None

def image_generator(image_urls, batch_size=5):
    """
    Generates batches of image data from a list of image URLs.

    Parameters:
    - image_urls (list of str): A list of image URLs to fetch.
    - batch_size (int): The number of images to fetch in each batch. Default is 25.

    Yields:
    - list of bytes: A batch of binary image data fetched from the URLs.
    """
    num_images = len(image_urls)
    start_index = 0
    while start_index < num_images:
        end_index = min(start_index + batch_size, num_images)
        batch_urls = image_urls[start_index:end_index]
        image_data_batch = [local_image_fetch(url.strip()) for url in batch_urls] # change local_image_fetch call to image_fetch if we
        yield image_data_batch
        start_index = end_index

def norm_banned_image_fetch(normal_images_path: str = None, banned_images_path: str = None, batch_size: int = 5):
    """
    Fetches image data for normal and banned images from text files.

    Parameters:
    - normal_images_path (str): The path to the text file containing URLs of normal images.
    - banned_images_path (str): The path to the text file containing URLs of banned images.
    - batch_size (int): The number of images to fetch in each batch.

    Returns:
    - tuple: A tuple containing two lists: normal_image_data and banned_image_data, each containing
      the fetched image data for normal and banned images, respectively.
    """
    normal_image_data = []
    banned_image_data = []

    if normal_images_path is not None:
        with open(normal_images_path, 'r') as file:
            normal_image_lines = file.readlines()
        normal_image_paths = extract_image_paths(normal_image_lines)
        normal_image_generator = image_generator(normal_image_paths, batch_size)
        normal_image_data.extend(next(normal_image_generator, []))

    if banned_images_path is not None:
        with open(banned_images_path, 'r') as file:
            banned_image_lines = file.readlines()
        banned_image_paths = extract_image_paths(banned_image_lines)
        banned_image_generator = image_generator(banned_image_paths, batch_size)
        banned_image_data.extend(next(banned_image_generator, []))

    normal_image_data_keras = [load_image_from_bytes(image) for image in normal_image_data]
    banned_image_data_keras = [load_image_from_bytes(image) for image in banned_image_data]
    normal_images = [image.tolist() for image in normal_image_data_keras]
    banned_images = [image.tolist() for image in banned_image_data_keras]

    return normal_images, banned_images
