from deeppavlov import build_model, configs,train_model, evaluate_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.ner.slotfill_dstc2)

my_config['metadata']['variables']['ROOT_PATH'] = 'C:/Users/user/Documents/GitHub/Intelligent-Chatbot/NLU/entity_extractor/slot_filler'

json.dump(my_config, open("slotfill_config.json","w"), indent=6)

model = build_model(my_config, download=True)

print(model(["any restaurent that offers vietnam food?"]))

print(evaluate_model(my_config))