    
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

trained_model_path = '../../model/'
trained_tokenizer = '../../tokenizer/'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

context ="Mary and John are a couple."
answer = "John"
text = "context: "+ context + " " + "answer: " + answer + " </s>"
print (text)

encoding = tokenizer.encode_plus(text, max_length = 64, padding=True, return_tensors="pt")
print (encoding.keys())
input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

model.eval()
beam_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=72,
    early_stopping=True,
    num_beams=5,
    num_return_sequences=3
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(sent)