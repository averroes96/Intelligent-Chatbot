from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
  
tokenizer = AutoTokenizer.from_pretrained("./translator/")
model = AutoModelForSeq2SeqLM.from_pretrained("./translator/")
dataset = load_dataset("empathetic_dialogues")

def translate(row):
    process_data(row)
    input_ids = tokenizer(row["prompt"], return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    row["prompt"] = decoded
    
    input_ids = tokenizer(row["utterance"], return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    row["utterance"] = decoded

    return row

def process_data(row):

    if len(row["prompt"].split(" ")) > 50:
        row["prompt"] = "<LONG TEXT>"
    if len(row["utterance"].split(" ")) > 50:
        row["utterance"] = "<LONG TEXT>"

dataset = dataset.map(translate)

dataset.save_to_disk("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/ds")