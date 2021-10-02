from deeppavlov import build_model, configs,train_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.ner.ner_dstc2)

my_config['metadata']['variables']["DATA_PATH"]= "../../../data/ner/"
my_config['metadata']['variables']["MODEL_PATH"]= "./model"
my_config["train"]["batch_size"] = 32
my_config["train"]["epochs"] = 200
my_config["train"]["validation_patience"] = 25
my_config["chainer"]["pipe"][3]["emb_dim"] = 128
my_config["chainer"]["pipe"][6]["learning_rate"] = 0.02
my_config["chainer"]["pipe"][6]["momentum"] = 0.95
my_config["chainer"]["pipe"][6]["n_hidden_list"] = [64, 64, 64]
my_config["train"]["evaluation_targets"] = ["train", "valid", "test"]

# json.dump(my_config, open("../task_oriented/entity_extractor/ner/config.json","w"), indent=6)

model = train_model(my_config)