from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import configs, train_model
from deeppavlov.metrics.accuracy import sets_accuracy
import json

model_config = read_json("config_file.json")

model_config["dataset_reader"]["data_path"] = "sample_data/FAQ.csv"
model_config["dataset_reader"]["data_url"] = None
config_file = open("config_file.json", "w")

faq = train_model(model_config)

json.dump(model_config, config_file, ensure_ascii = True, indent=6)

faq.save()

from deeppavlov import build_model
from deeppavlov.metrics.accuracy import sets_accuracy

model_config = read_json("config_file.json")

faq = build_model(model_config)

result = faq(['من هو أفضل لاعب في العالم؟'])

answer = "عذرا لم أفهم سؤالك !"

for vals in result[1]:
  for val in vals:
    if val > 0.75:
      answer = result[0]