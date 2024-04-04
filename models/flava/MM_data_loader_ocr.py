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

def get_meme_text(image):
    config = "-l eng+chi_sim+chi_tra+tam+msa --psm 4 --oem 1"

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

def get_meme_text(image):
    config = "-l eng+chi_sim+chi_tra+tam+msa --psm 4 --oem 1"
    #config = "-l chi_tra --psm 4 --oem 1"

    text = pytesseract.image_to_string(image, config)

    text = pytesseract.image_to_string(image, config=config)
    d = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
    n_boxes = len(d["level"])
    coordinates = []

    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        coordinates.append((x, y, w, h))
    return text, coordinates

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
            vals = ast.literal_eval(self.ocr[file_name[4:]])
            ocr_text = ""
            for item in vals:
                ocr_text +=  ". " + item[1] # 1st item is OCR text 
                bbox.append(item[0]) # 0th item is bounding box of the text
            #ic("Dev",file_name,ocr_text)

        # Option2 Use Ground Truth
        elif self.split=='train':
            ocr_text = self.lb[idx]["text"]
            #ic("Train",file_name,ocr_text)
        else:
            ocr_text = ""
        

        # Apply mask and inpainting , inputs -> bounding boxes
        # for coordinates in bbox:
        #     # Get image mask for image inpainting
        #     im_mask = get_image_mask(image=image, coordinates_to_mask=coordinates)

        #     # cv2.imwrite("/tmp/temp_image_mask.png", im_mask)
        #     # # (OPTIONAL) Read from /tmp folder
        #     # im_mask = cv2.imread("/tmp/temp_image_mask.png", cv2.IMREAD_GRAYSCALE)

        #     # Perform image inpainting
        #     image_inpainted = get_image_inpainted(image=image, image_mask=im_mask) 
                
        #     image = image_inpainted
            
        #     ic("inpaint done")

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




