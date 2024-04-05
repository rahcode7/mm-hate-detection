# Input of FB images at train and test time 
import easyocr
import os 
import json 
import torch
import numpy as np 
from tqdm import tqdm 
from icecream import ic 

if __name__ == "__main__":
    
    # Read images
    IMG_PATH = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data/img" 
    OUTPUT_PATH = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/results/FB-HM"
    #01235.png
    
    # detect language - English, singlish etc
    reader = easyocr.Reader(['en'])

    #d = {}
    cnt = 0 
    for image in tqdm(os.listdir((IMG_PATH))):
        cnt +=1 
        if cnt < 1077:
            continue 
        else:
            with open(os.path.join(OUTPUT_PATH,'ocr-fbhm.json'),'a+') as f:
                image_path = os.path.join(IMG_PATH,image)
                # English defautls
                result = reader.readtext(image_path)
                #ic(result,image)
                # store in image : list
                d = {}
                d[image] = str(result)
                #ic(d)
                json.dump(d,f)
                f.write('\n')
                f.close()

    




