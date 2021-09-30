from transformers import BertModel,BertTokenizer
from deeppavlov.core.common.file import read_json
import sys
import json

project_configs = read_json("configs.json")

if len(sys.argv) == 3:
    project_configs["BERT"]["MODEL"] = sys.argv[1]
    project_configs["BERT"]["SAVE_PATH"] = sys.argv[2]

if len(sys.argv) == 2:
    project_configs["BERT"]["MODEL"] = sys.argv[1]

tokenizer = BertTokenizer.from_pretrained(project_configs["BERT"]["MODEL"])
model = BertModel.from_pretrained(project_configs["BERT"]["MODEL"])

tokenizer.save_pretrained(project_configs["BERT"]["SAVE_PATH"])
model.save_pretrained(project_configs["BERT"]["SAVE_PATH"])

json.dump(project_configs, open("configs.json", "w"), indent=6)

print(f'MODEL {project_configs["BERT"]["MODEL"]} WAS SUCCESSFULY SAVED AT {project_configs["BERT"]["SAVE_PATH"]} !')