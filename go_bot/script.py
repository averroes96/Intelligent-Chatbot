from deeppavlov import configs, build_model
from deeppavlov.core.common.file import read_json
import json

gobot_config = read_json(configs.go_bot.gobot_dstc2_best_json_nlg)

gobot_config["chainer"]["pipe"][2]["id"] = "hsfashion_database" # Database ID
gobot_config["chainer"]["pipe"][2]["save_path"] = "{DOWNLOADS_PATH}/dstc2/hsf.sqlite" # DB save path
gobot_config["chainer"]["pipe"][3]["database"] = "#hsfashion_database" # DB reference
gobot_config["chainer"]["pipe"][3]["intent_classifier"]["config_path"] = "{CONFIGS_PATH}/classifiers/intents_dstc2_bert.json" # Intent Classifier configuration
gobot_config['metadata']['variables']['ROOT_PATH'] = 'C:/Users/user/Documents/GitHub/Intelligent-Chatbot/GO_BOT'

json.dump(gobot_config, open("gobot_config.json", "w"), indent=6)

chatbot = build_model(gobot_config, download=True)

user_text = input.strip().lower()

while user_text != "bye":
    chatbot([user_text])
    user_text = input.strip().lower()