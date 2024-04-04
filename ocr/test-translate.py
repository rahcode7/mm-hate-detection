from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
#text = "天若有情天亦老 葡式蛋挞配腿堡"
text = "சுகருக்காக நடற்தவன விட"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# # translate Hindi to French
# tokenizer.src_lang = "hi"
# encoded_hi = tokenizer(hi_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
# print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# translate Chinese to English
tokenizer.src_lang = "ta"
encoded_zh = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))

print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))