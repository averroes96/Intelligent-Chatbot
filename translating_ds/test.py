from datasets import load_dataset, load_from_disk
from datasets import Dataset

dataset = load_from_disk("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/ds/test")

dataset.to_csv("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/temp/test.csv")

dataset = load_from_disk("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/ds/validation")

dataset.to_csv("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/temp/valid.csv")

