from deeppavlov import build_model, configs,train_model, evaluate_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.ner.slotfill_dstc2)

my_config['metadata']['variables']['NER_CONFIG_PATH'] = '../ner/ner_config.json'
my_config['metadata']['variables']["MODEL_PATH"]= "./model/"
my_config['metadata']['variables']["SLOT_VALS_PATH"]= "../../../data/dstc2/dstc_slot_vals.json"

json.dump(my_config, open("slotfill_config.json","w"), indent=6)

model = build_model(my_config)

utterance = input().strip()

while utterance != "quit":
    print(model([utterance]))
    utterance = input().strip()
