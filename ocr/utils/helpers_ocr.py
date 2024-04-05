import pytesseract
from pytesseract import Output
import cv2
from icecream import ic 
from langdetect import detect
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import numpy as np
import cv2 


def get_meme_text_easyocr(image):
    reader = easyocr.Reader(['en','ch_sim'])
    result = reader.readtext(filepath)
    vals = ast.literal_eval(self.ocr[file_name[4:]])
    ocr_text = ""
    for item in vals:
        ocr_text +=  ". " + item[1] # 1st item is OCR text 
        bbox.append(item[0]) # 0th item is bounding box of the text
    return result

def get_meme_text_pytesseract(image):
    config = "-l eng+chi_sim+chi_tra+tam+msa --psm 4 --oem 1"

    text = pytesseract.image_to_string(image, config)

    text = pytesseract.image_to_string(image, config=config)
    d = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
    n_boxes = len(d["level"])
    coordinates = []

    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        coordinates.append((x, y, w, h))
    return text, coordinates

def get_image_mask(image, coordinates_to_mask):
    # Create a mask image with image_size
    image_mask = np.zeros_like(image[:, :, 0])

    for coordinates in coordinates_to_mask:
        # unpack the coordinates
        x, y, w, h = coordinates

        # set mask to 255 for coordinates
        image_mask[y : y + h, x : x + w] = 255

    return image_mask

def get_image_inpainted(image, image_mask):
    # Perform image inpainting to remove text from the original image
    image_inpainted = cv2.inpaint(
        image, image_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    return image_inpainted
