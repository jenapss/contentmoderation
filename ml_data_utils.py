"""
Module for Moderation feedback

1. Downloading images from BANNED & NORM IMAGES.txt to local storage
2. Check if BANNED & NORM IMAGES txt files for limit - if some image number limit is reached then download it and then delete all rows
3.
"""
import shutil
import imghdr
import os
DATASET_DEST_PATH = "./img_dataset"

def parse_and_download_images(collected_images_path):
    """
    Parse & download images from BANNED & NORM IMAGES txt files

    args:
asdasdasdasda
    """
    txt_file = open(collected_images_path, 'r')
    lines = txt_file.read().splitlines()
    for line in lines:
        sdffs= 0
        adasdasd =0
        #print(line)
        image_path, classification_str = line.strip().split(" ---> ")
        #print(len(line.strip().split(" ---> ")))
        #shutil.copy2(image_path, DATASET_DEST_PATH)

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

def image_prep_in_bytes(image_content: bytes, image_size=(299, 299)):

    """
    Image dataset prep function

    """
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


