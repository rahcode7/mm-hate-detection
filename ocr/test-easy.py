
import cv2
from icecream import ic 
from langdetect import detect
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import easyocr


if __name__ == "__main__":

    #filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/AISG-Online-Safety-Challenge-Submission-Guide/local_test/test_input/8b52fi.png"
    filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/chinese.jpeg"
    #filepath ="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/tamil.jpeg"
    #filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/Tamil_troll_memes/test_img/test_img_0.jpg"
    #filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/malay.jpg"
    #filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/malay2.jpg"
    #filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data/img/72048.png"
    # 1. Open image filepath ========================================= #


    reader = easyocr.Reader(['hi','en'],gpu=False) #,'ta','ms','hi','id'])
    # reader = easyocr.Reader(['ta','en'])
    result = reader.readtext(filepath)
    ic(result)

    ocr_text = ""
    for item in result:
        ocr_text +=  " " + item[1]
    ic(ocr_text)
   
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    detected_lang = detect(ocr_text)

    if detected_lang in  ['ko','ja','zh']:
        ic(detected_lang)
        # Chinese to english translate
        tokenizer.src_lang = "zh"
        encoded_zh = tokenizer(ocr_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
        ocr_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ic(ocr_text)

   