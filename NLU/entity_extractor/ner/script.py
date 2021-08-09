from deeppavlov import build_model, configs,train_model
from deeppavlov.core.common.file import read_json
import json

my_config = read_json(configs.ner.ner_ontonotes_bert)

my_config['metadata']['variables']['ROOT_PATH'] = 'C:/Users/user/Documents/GitHub/Intelligent-Chatbot/NLU/entity_extractor/ner'
my_config['dataset_reader']['data_path']= '{ROOT_PATH}/dataset/'
my_config['metadata']['variables']["NER_PATH"]= "{MODELS_PATH}/ner_bert"
my_config['metadata']['variables']["BERT_PATH"]= "C:/Users/user/Documents/GitHub/Intelligent-Chatbot/bert_models/cased_L-12_H-768_A-12"
my_config['chainer']['pipe'][1]["save_path"]= '{NER_PATH}/tag.dict'
my_config['chainer']['pipe'][1]["load_path"]= '{NER_PATH}/tag.dict'
my_config['chainer']['pipe'][2]["save_path"]= '{NER_PATH}/model'
my_config['chainer']['pipe'][2]["load_path"]= '{NER_PATH}/model'

json.dump(my_config, open("ner_bert_config.json","w"), indent=6)

model = train_model(my_config, download=True)