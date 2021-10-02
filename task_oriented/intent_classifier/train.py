from deeppavlov import build_model, configs, train_model, evaluate_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.classifiers.intents_dstc2)

my_config['dataset_reader']['data_path'] = "../../data/dstc2/"
my_config['metadata']['variables']['MODEL_PATH'] = "./model"
my_config['chainer']['pipe'][2]['load_path'] = "../../data/embedder/cc.fr.100.bin"
my_config['chainer']['pipe'][4]['learning_rate_decay'] = 0.01
my_config['chainer']['pipe'][4]['dropout_rate'] = 0.1
my_config['train']["validation_patience"] = 50
my_config['train']["batch_size"] = 16

model = train_model(my_config)