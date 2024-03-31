from transformers import AutoTokenizer, FlavaMultimodalModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
model = FlavaMultimodalModel.from_pretrained("facebook/flava-full")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)