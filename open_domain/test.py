from transformers import AutoTokenizer
from transformers import EncoderDecoderModel

LOAD_PATH = "./chitchat"

tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH)
model = EncoderDecoderModel.from_pretrained(LOAD_PATH)

user_utter = input().strip()

while user_utter != "quit":
    input_ids = tokenizer(user_utter, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"> {decoded}")
    user_utter = input().strip()
