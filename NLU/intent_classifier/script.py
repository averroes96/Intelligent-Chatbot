from deeppavlov import build_model, configs
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.classifiers.intents_dstc2_bert)

my_config['metadata']['variables']['ROOT_PATH'] = "C:/Users/user/Documents/GitHub/Intelligent-Chatbot/NLU/intent_classifier"

json.dump(my_config, open("intent_classifier_config.json","w"), indent=6)

model = build_model(my_config)

print("=" * 48)

user_utter = input().strip()

while user_utter != "quit":
    print(model([user_utter])[0])
    if model([user_utter])[1] == False:
        print("No intent was recognised")
    else:
        print(model([user_utter])[1])

    user_utter = input().strip()