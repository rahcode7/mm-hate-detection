# Standard Python Library
import sys
import logging
import torch 
from transformers import BertTokenizer
import numpy as np
from icecream import ic  
import torch.nn as nn
from transformers import AutoTokenizer, FlavaMultimodalModel, AutoProcessor,FlavaModel
import torch
import numpy as np
from PIL import Image
import os 
import easyocr
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import shutil
def get_meme_text(filepath,model,tokenizer,translate=False):
    result = reader.readtext(filepath)

    ocr_text = ""
    for item in result:
        ocr_text +=  " " + item[1]
    ic(ocr_text)

    if translate:
        detected_lang = detect(ocr_text)

    if  detected_lang in  ['ko','ja','zh']:
        #ic(detected_lang)
        # Chinese to english translate
        tokenizer.src_lang = "zh"
        encoded_zh = tokenizer(ocr_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
        ocr_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ic(ocr_text)

    return ocr_text

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
    checkpoint = torch.load(checkpoint_file,map_location=device)
    model.load_state_dict(checkpoint['model'],strict=False)
    model = model.to(device)
    model.eval()
    #ic("Checkpoint loaded")


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

    #ic(probs,predicted)
    return probs, predicted
    #return "",""

def process_line_by_line(checkpoint_file,reader,device,model_dict,image_path):
    print("Processing image")
    image = Image.open(image_path).convert('RGB')  
    
    text = get_meme_text(reader,image_path,translation_model,tokenizer,translate=True) # Easyocr
    proba, label = classifier(model_dict,checkpoint_file,device,image=image, text=text)

    return proba, label

if __name__ == "__main__":

    os.environ['TRANSFORMERS_CACHE'] = 'cache'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_file="checkpoints/checkpoint_05.pth"

    # Load model artifacts 

    PATH='cache/models--facebook--flava-full'
    tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    model = FlavaModel.from_pretrained(PATH, local_files_only=True)
    processor = AutoProcessor.from_pretrained(PATH,local_files_only=True)

    # tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full",PATH, local_files_only=True)
    # model = FlavaModel.from_pretrained("facebook/flava-full",PATH, local_files_only=True)
    # processor = AutoProcessor.from_pretrained("facebook/flava-full",PATH, local_files_only=True)
    model_dict = {'tokenizer':tokenizer,'model':model,'processor':processor}

    # Text reader
    # print(os.getcwd())
    # print(os.listdir())
    # #shutil.chown('home/admin/.EasyOCR', user='admin', group=None)
    # shutil.chown('/home/admin/app/.EasyOCR', user='admin', group=None)

    EASYOCR_PATH="cache" 
    reader = easyocr.Reader(['ch_sim','en'],model_storage_directory=EASYOCR_PATH,download_enabled=False,user_network_directory=EASYOCR_PATH,gpu=False) #,quantize=False) #,'ta','ms','hi','id'])
    
    # Translation model load
    TRANSLATION_MODEL='cache/models--facebook--m2m100_418M'
    translation_model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL,local_files_only=True)
    tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL,local_files_only=True)
    print("models loaded" )
    #ic(os.getcwd())
    # Iteration loop to get new image filepath from sys.stdin:
    for line in sys.stdin:
        # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
        image_path = line.rstrip()
        #print(image_path)
        try:
            # Process the image
            proba, label = process_line_by_line(checkpoint_file,reader,device,model_dict,image_path)

            # Ensure each result for each image_path is a new line
            sys.stdout.write(f"{proba:.4f}\t{label}\n")

        except Exception as e:
            # Output to any raised/caught error/exceptions to stderr
            sys.stderr.write(str(e))
