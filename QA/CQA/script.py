from deeppavlov import build_model, configs, train_model
from deeppavlov.core.common.file import read_json
import json

model_config = read_json(configs.squad.squad_bert)

model_config["metadata"]["variables"]["ROOT_PATH"] = "C:/Users/user/Documents/GitHub/Intelligent-Chatbot/QA/CQA"
json.dump(model_config, open("cqa_config.json","w"), indent=6)

cqa_model = build_model(model_config, download=True)

context = """H.S.Fashion is a brand company specialized in selling both women and men shoes located in chlef center. 
H.S.Fashion sells in retails and in wholesale through its points of sell which are based in Boulevard des Martyrs Chlef, Algérie for retail selling and Zone d'activités Ouled Mohamed, Chlef, Algérie for wholesale selling. 
Our products are all imported mainly from China and Spain."""

user_utter = input().strip()

while user_utter != "quit":
    print(cqa_model([context],[user_utter]))
    user_utter = input().strip()