import fasttext
from deeppavlov.core.common.file import read_json
import sys
import json

project_configs = read_json("configs.json")

if len(sys.argv) == 3:
    project_configs["FASTTEXT"]["EMBEDDER"] = sys.argv[1]
    project_configs["FASTTEXT"]["SAVE_PATH"] = sys.argv[2]

if len(sys.argv) == 2:
    project_configs["FASTTEXT"]["EMBEDDER"] = sys.argv[1]

ft = fasttext.load_model(project_configs["FASTTEXT"]["EMBEDDER"])
fasttext.util.reduce_model(ft, 100)

json.dump(project_configs, open("configs.json", "w"), indent=6)

ft.save_model(project_configs["FASTTEXT"]["SAVE_PATH"])

print(f'MODEL {project_configs["FASTTEXT"]["EMBEDDER"]} WAS SUCCESSFULY SAVED AT {project_configs["FASTTEXT"]["SAVE_PATH"]} !')