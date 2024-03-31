# Standard Python Library
import sys
import logging
import torch 
import easyocr
from transformers import BertTokenizer
import numpy as np
from icecream import ic 
import argparse
import wandb
import os 
from torch.utils.data import DataLoader
from MM_data_loader_ocr import FBHMDataset,collate_fn
import json 
import shutil
import site
import torch.nn as nn
from transformers import AutoTokenizer, FlavaMultimodalModel, AutoProcessor,FlavaModel
import torch
import numpy as np
from PIL import Image

# def get_meme_text(reader,image_path):
#     l = reader.readtext(image_path)
#     ic(l)
#     text = ""
#     for item in l:
#         text += ". " + item[1]
#     ic(text)
#     # Later - Add coordinates to return for inpainting 
#     return text

# Pytesseract
import pytesseract
from pytesseract import Output
import cv2
def get_meme_text_pyt(image_path):
    image = cv2.imread(image_path)

    config = "-l eng+chi_sim+chi_tra+tam+msa --psm 4 --oem 1"

    text = pytesseract.image_to_string(image, config=config)
    ic(text)
    
    d = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
    n_boxes = len(d["level"])
    coordinates = []

    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        coordinates.append((x, y, w, h))

    return text
   
class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def classifier(model_dict,checkpoint_file,device,image,text):
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    processor = model_dict['processor']

    # Read all image into dataloader 
    # train_dataset = FBHMDataset(root_dir=TEST_DIR,split='test')
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn= collate_fn,pin_memory=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'],strict=False)
    model = model.to(device)
    model.eval()
    ic("Checkpoint loaded")


    # Inferece only
    inputs = processor(text = text, images=image,return_tensors="pt",padding=True)
    inputs = inputs.to(device)

    outputs = model(**inputs)
    #ic(outputs)s

    classification_head = ClassificationHead(768,num_classes=1)
    classification_head = classification_head.to(device)
    classifier_inputs = outputs.multimodal_output.pooler_output.to(device)
    logits = classification_head(classifier_inputs).cpu()

    probs = torch.sigmoid(logits)
    probs = probs.squeeze()
    predicted = np.where(probs >0.5,1,0).tolist()
    #predicted = torch.Tensor(predicted.squeeze().tolist()) # .to(device)

    ic(probs,predicted)
    return probs, predicted
    #return "",""

def process_line_by_line(checkpoint_file,reader,device,model_dict,image_path):
    ic("Processing image")
    image = Image.open(image_path).convert('RGB')  
    
    text = get_meme_text(reader,image_path) # Easyocr
    #text = get_meme_text_pyt(image_path) # Pyt 

    proba, label = classifier(model_dict,checkpoint_file,device,image=image, text=text)

    return proba, label

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_file="checkpoints/checkpoints-flava-ddp-base-29Mar/checkpoint_04.pth"

    # Load model artifacts 
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
    model = FlavaModel.from_pretrained("facebook/flava-full")
    processor = AutoProcessor.from_pretrained("facebook/flava-full")
    model_dict = {'tokenizer':tokenizer,'model':model,'processor':processor}

    ic("Add Input images")
    # Iteration loop to get new image filepath from sys.stdin:
    for line in sys.stdin:
        # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
        image_path = line.rstrip()
        ic(image_path)
        #try:
            # Process the image
        proba, label = process_line_by_line(checkpoint_file,reader,device,model_dict,image_path)

        # Ensure each result for each image_path is a new line
        sys.stdout.write(f"{proba:.4f}\t{label}\n")

        # except Exception as e:
        #     # Output to any raised/caught error/exceptions to stderr
        #     sys.stderr.write(str(e))
