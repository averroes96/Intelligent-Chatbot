from deeppavlov import build_model, configs
from deeppavlov.core.common.file import read_json
import json

def save_model():
    model_config = read_json("config_file.json")
    model_qa = build_model(model_config)
    # config_file = open("config_file.json", "w")
    # json.dump(model_config, config_file, ensure_ascii = True, indent=6)
    model_qa.save()

def load_model():
    model_config = read_json("/content/config_file.json")
    qa_model = build_model(model_config)

    return qa_model