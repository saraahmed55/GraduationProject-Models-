import requests
from PIL import Image 
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering 
import torch

#image with Ques
url = "http://images.cocodataset.org/test-stuff2017/000000002558.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "What color of traffic signal?"


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
encoding = processor(image, text, return_tensors="pt")
  

#load the ViLT model
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

outputs = model(**encoding)
logits = outputs.logits
idx = torch.sigmoid(logits).argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
     
  
  
# Requirements 
# download a "!pip install -q git+https://github.com/huggingface/transformers.git"
     
