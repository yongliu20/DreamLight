import os, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image 


def resize_shorter(image, target_size=1024):
    pil_image = image#Image.fromarray(image)
    original_width, original_height = pil_image.size
    size = min(original_width, original_height)
    ratio = target_size/size 
    resized_width = int(ratio*original_width)
    resized_height = int(ratio*original_height)

    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    return resized_image


def check_directory(dir):
    if os.path.exists(dir) == False:
        os.mkdir(dir) 


