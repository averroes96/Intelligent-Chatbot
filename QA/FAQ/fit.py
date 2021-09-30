from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import configs, train_model
import json

model_config = read_json(configs.faq.tfidf_logreg_en_faq)
model_config["dataset_reader"]["data_path"] = "../../data/QA.csv"
model_config["dataset_reader"]["data_url"] = None
model_config["metadata"]["variables"]["MODEL_PATH"] = "./model"
model_config["chainer"]["pipe"][2]["save_path"] = "{MODEL_PATH}/tfidf.pkl"
model_config["chainer"]["pipe"][2]["load_path"] = "{MODEL_PATH}/tfidf.pkl"
model_config["chainer"]["pipe"][3]["save_path"] = "{MODEL_PATH}/tfidf_vec.pkl"
model_config["chainer"]["pipe"][3]["load_path"] = "{MODEL_PATH}/tfidf_vec.pkl"
model_config["chainer"]["pipe"][4]["save_path"] = "{MODEL_PATH}/answers_vocab.pkl"
model_config["chainer"]["pipe"][4]["load_path"] = "{MODEL_PATH}/answers_vocab.pkl"
model_config["chainer"]["pipe"][5]["save_path"] = "{MODEL_PATH}/logreg.pkl"
model_config["chainer"]["pipe"][5]["load_path"] = "{MODEL_PATH}/logreg.pkl"

faq = train_model(model_config)

json.dump(model_config, open("config.json","w"), indent=6)