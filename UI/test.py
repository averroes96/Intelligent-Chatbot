from transformers import CamembertTokenizer, CamembertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import os
from datetime import datetime

# def get_sentence_emb(sents):
#   tokenized_sentence = tokenizer.tokenize(sents)
#   encoded_sentence = tokenizer.encode(tokenized_sentence)
#   encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
#   embeddings = camembert(encoded_sentence)[1]
#   return embeddings

# # You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
# tokenizer = CamembertTokenizer.from_pretrained("../models/camembert/")
# camembert = CamembertModel.from_pretrained("../models/camembert/")

# data = pd.read_csv("../data/classifier_data.csv")

# data["class"] = data['class'].map({'QA': [1,0,0], 'task': [0,1,0], "chitchat" : [0,0,1]})
# embeddings = []

# for index, row in data.iterrows():
#   start = datetime.now()
#   emb = get_sentence_emb(row["text"])
#   end = datetime.now()
#   elapsed = end - start
#   left = elapsed * (len(data) - (index + 1))
#   print(f"{(index/len(data)) * 100} ... TIME LEFT = {left}")
#   embeddings.append([emb.detach().numpy(), row["class"]])


CONTEXTS_PATH = "../data/cqa_contexts/"

def get_all_contexts(path):
    files = os.listdir(path)
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)

get_all_contexts(CONTEXTS_PATH)