# Input of FB images at train and test time 
import pytesseract
import os 
import json 
import torch
import numpy as np 
from tqdm import tqdm 
from icecream import ic 
import cv2
from utils.helpers_ocr import get_meme_text_pytesseract,get_image_mask,get_image_inpainted
import easyocr

if __name__ == "__main__":
    
    # Read images
    IMG_OUTPUT_PATH = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data/img-inpainted-dev"
    
    PATH="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data"

    #filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/malay2.jpg"

    # Get dev file in dict
    FILE_PATH = os.path.join(PATH,'dev.jsonl')
    data_ocr = []
    with open(FILE_PATH,'r',encoding='utf8') as f:
        for line in f:
            data_ocr.append(json.loads(line))
    ic(data_ocr[-1])

    # single_dict = {}
    # for d in data_ocr:
    #     single_dict.update(d)
    # ic(single_dict)
    image_list = []
    for d in data_ocr:
        image_list.append(d['img'])
    ic(image_list[0:5])


    reader = easyocr.Reader(['en','ch_sim'])

    # image_list = ["/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/chinese.jpeg"]
    cnt=0
    for image_name in tqdm(image_list):
        cnt+=1
        if cnt>=5:
            break

        # Extract text
        with open(os.path.join(PATH,'ocr-fbhm-pyt-dev.json'),'a+') as f:
            image_path = os.path.join(PATH,image_name)
            im = cv2.imread(image_path)
            text, coordinates = get_meme_text_pytesseract(image=im)

            result = reader.readtext(image_path)
            ic(result)
            
            d = {}
            d[image_name[4:]] = str(text)
            ic(text,d)
            json.dump(d,f)
            f.write('\n')

            # Add image inpainting 
            # ic("Image painting")
            # im_mask = get_image_mask(image=im, coordinates_to_mask=coordinates)
            # im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)

            # op_path=os.path.join(IMG_OUTPUT_PATH,image_name[4:])
            # ic(op_path)
            # cv2.imwrite(op_path,im_inpainted)

            f.close()

        



