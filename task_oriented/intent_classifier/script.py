from deeppavlov import build_model, configs, train_model
from deeppavlov.core.common.file import read_json
import json

# my_config = read_json(configs.classifiers.intents_dstc2)

# my_config['dataset_reader']['data_path'] = "./dataset/"
# my_config['metadata']['variables']['MODEL_PATH'] = "./model"
# my_config['chainer']['pipe'][2]['load_path'] = "../embedders/cc.en.100.bin"
# my_config['train']["validation_patience"] = 100
# my_config['metadata']['download'][0]["url"] = "http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip"
# my_config['chainer']['pipe'][1]["vocab_file"] = "{DOWNLOADS_PATH}/bert_models/multi_cased_L-12_H-768_A-12/vocab.txt"
# my_config['chainer']['pipe'][3]["bert_config_file"] = "{DOWNLOADS_PATH}/bert_models/multi_cased_L-12_H-768_A-12/bert_config.json"
# my_config['chainer']['pipe'][3]["pretrained_bert"] = "{DOWNLOADS_PATH}/bert_models/multi_cased_L-12_H-768_A-12/bert_model.ckpt"

# json.dump(my_config, open("intent_classifier_config.json","w"), indent=6)

# model = train_model(my_config)

my_config = read_json("intent_classifier_config.json")

model = build_model(my_config)

print("=" * 48)

user_utter = input().strip()

while user_utter != "quit":
    print(model([user_utter]))
    user_utter = input().strip()