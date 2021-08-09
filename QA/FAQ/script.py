from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import configs, train_model
import json

model_config = read_json(configs.faq.tfidf_logreg_en_faq)
model_config["dataset_reader"]["data_path"] = "FAQ.csv"
model_config["dataset_reader"]["data_url"] = None
model_config["metadata"]["variables"]["LOCAL_PATH"] = "C:/Users/user/Documents/GitHub/Intelligent-Chatbot"
model_config["metadata"]["variables"]["FAQ_MODEL_PATH"] = "{LOCAL_PATH}/QA/faq/model"
model_config["chainer"]["pipe"][2]["save_path"] = "{FAQ_MODEL_PATH}/tfidf_vec.pkl"
model_config["chainer"]["pipe"][2]["load_path"] = "{FAQ_MODEL_PATH}/tfidf_vec.pkl"
model_config["chainer"]["pipe"][3]["save_path"] = "{FAQ_MODEL_PATH}/answers_vocab.pkl"
model_config["chainer"]["pipe"][3]["load_path"] = "{FAQ_MODEL_PATH}/answers_vocab.pkl"
model_config["chainer"]["pipe"][4]["save_path"] = "{FAQ_MODEL_PATH}/logreg.pkl"
model_config["chainer"]["pipe"][4]["load_path"] = "{FAQ_MODEL_PATH}/logreg.pkl"

faq = train_model(model_config)

json.dump(model_config, open("faq_config.json","w"), indent=6)

model_config = read_json("faq_config.json")

faq_model = build_model(model_config)

user_utter = input().strip()

while user_utter != "quit":
    print(faq_model([user_utter]))
    user_utter = input().strip()