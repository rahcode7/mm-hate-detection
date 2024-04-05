## Load FHM dataset
from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image
import json 
from icecream import ic 
import ast 
import re
import cv2
import numpy as np 
import pytesseract
from pytesseract import Output
from utils.helpers_ocr import get_meme_text_pytesseract,get_image_mask,get_image_inpainted



class FBHMDataset(Dataset):
    def __init__(self,root_dir,split) -> None:
        splitfile = split + '.jsonl'
        data = []
        
        FILE_PATH = os.path.join(root_dir,splitfile)
        with open(FILE_PATH,'r',encoding='utf8') as f:
            for line in f:
                data.append(json.loads(line))
        self.lb = data

        # Ocr file
        FILE_PATH = os.path.join(root_dir,'ocr-fbhm.json')
        data_ocr = []
        with open(FILE_PATH,'r',encoding='utf8') as f:
            for line in f:
                data_ocr.append(json.loads(line))

        single_dict = {}
        for d in data_ocr:
            single_dict.update(d)

        self.ocr = single_dict
        self.split = split

        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.lb)

    def __getitem__(self,idx):
        file_name = str(self.lb[idx]["img"]) # + ".png"     
        label = self.lb[idx]["label"]
        
        
        image_path = os.path.join(self.root_dir,file_name)
        image = Image.open(image_path).convert('RGB')  

        bbox = []
        if self.split=='dev':

            # For EasyOCR
            # vals = ast.literal_eval(self.ocr[file_name[4:]])
            # ocr_text = ""
            # for item in vals:
            #     ocr_text +=  ". " + item[1] # 1st item is OCR text 
            #     bbox.append(item[0]) # 0th item is bounding box of the text
            #ic("Dev",file_name,ocr_text)

            # For pytesseract
            im = cv2.imread(image_path)
            text, coordinates = get_meme_text_pytesseract(image=im)

            # Image inpainting
            im_mask = get_image_mask(image=im, coordinates_to_mask=coordinates)
            im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)
            image =im_inpainted
            ic("inpaint done")


        # Use Ground Truth for train
        elif self.split=='train':
            ocr_text = self.lb[idx]["text"]
            #ic("Train",file_name,ocr_text)

            # For pytesseract
            # im = cv2.imread(image_path)
            # text, coordinates = get_meme_text(image=im)
        else:
            ocr_text = ""

        # keep alphanum
        ocr_text  = re.sub(r'\W+', ' ', ocr_text).strip()
        #ic(file_name,image_path,ocr_text,label)

        return file_name,image,ocr_text,label

def collate_fn(batch):
    file_list,image_list,text_list,label_list = [],[],[],[]

    batch = list(filter (lambda x:x is not None, batch))
    
    for file,image,ocr_text,label in batch:
        file_list.append(file)
        image_list.append(image)
        text_list.append(ocr_text)
        label_list.append(label)    
    return file_list,image_list,text_list,label_list




