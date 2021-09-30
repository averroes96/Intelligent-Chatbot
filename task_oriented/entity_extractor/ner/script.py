from deeppavlov import build_model, configs,train_model, evaluate_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.ner.ner_dstc2)

# my_config['metadata']['variables']["ROOT_PATH"]= "C:/Users/user/Documents/GitHub/Intelligent-Chatbot/NLU/entity_extractor/ner"
my_config['metadata']['variables']["DATA_PATH"]= "./dataset"
my_config['metadata']['variables']["MODEL_PATH"]= "./model"
# my_config['metadata']['variables']["MODELS_PATH"]= "./models"
# my_config['metadata']['variables']["SLOT_VALS_PATH"]= "./dataset/dstc_slot_vals.json"
# my_config['chainer']['pipe'][2]["save_path"]= '{MODEL_PATH}/tag.dict'
# my_config['chainer']['pipe'][2]["load_path"]= '{MODEL_PATH}/tag.dict'
# my_config['chainer']['pipe'][2]["save_path"]= '{NER_PATH}/model'
# my_config['chainer']['pipe'][2]["load_path"]= '{NER_PATH}/model'

json.dump(my_config, open("ner_config.json","w"), indent=6)

# my_config = read_json("ner_config.json")

# model = build_model(my_config)

# utterance = input().strip()

# print(f"eval = {evaluate_model(my_config)}")

# while utterance != "quit":
#     print(model([utterance]))
#     utterance = input().strip()
# print(model(["I want a cheap restaurent in the west part of town that serves indiane food"]))