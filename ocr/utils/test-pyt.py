import pytesseract
from pytesseract import Output
import cv2
from icecream import ic 
from langdetect import detect
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


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
    #return text,text

if __name__ == "__main__":

    #filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/AISG-Online-Safety-Challenge-Submission-Guide/local_test/test_input/8b52fi.png"
    #filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/chinese.jpeg"
    #filepath ="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/tamil.jpeg"
    #filepath = "/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/Tamil_troll_memes/test_img/test_img_0.jpg"
    #filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/malay.jpg"
    filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/SG/malay2.jpg"
    filepath="/Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data/img/72048.png"
    # 1. Open image filepath ========================================= #
    im = cv2.imread(filepath)
    
    # 2. Get meme text =============================================== #
    text, coordinates = get_meme_text(image=im)
    ic(text)


    


    # Multilingual data -> output English text
    # Step1 Detect script
    # osd = pytesseract.image_to_osd(filepath,config='--psm 0 -c min_characters_to_try=5')
    # script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
    # ic(script)

    # script_dict = {'tamil':'ta','chinese':'zh','japanese':'zh','hindi':'hi','malayalam':'ma'}

    # Translate to English
    # Step 2 Translate to english
    # from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    # model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # tokenizer.src_lang = "ma"
    # encoded_zh = tokenizer(text, return_tensors="pt")
    # ocr_text = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))

    # ic(text,coordinates)
