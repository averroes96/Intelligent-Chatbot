from deeppavlov import configs, train_model,build_model,evaluate_model
from deeppavlov.core.common.file import read_json

to_config = read_json(configs.go_bot.gobot_dstc2_best)

to_config["dataset_reader"]["data_path"] = "../data/dstc2/"
to_config["chainer"]["pipe"][1]["save_path"] = "./bot/word.dict"
to_config["chainer"]["pipe"][1]["load_path"] = "./bot/word.dict"
to_config["chainer"]["pipe"][2]["id"] = "hsfashion_database" # Database ID
to_config["chainer"]["pipe"][2]["table_name"] = "main_table" # Database table
to_config["chainer"]["pipe"][2]["save_path"] = "../data/dstc2/hsfashion.sqlite" # DB save path
to_config["chainer"]["pipe"][3]["database"] = "#hsfashion_database" # DB reference
to_config["chainer"]["pipe"][3]["intent_classifier"]["config_path"] = "./intent_classifier/intent_classifier_config.json" # Intent Classifier configuration
to_config["chainer"]["pipe"][3]["slot_filler"]["config_path"] = "./entity_extractor/slot_filler/slotfill_config.json"
to_config["chainer"]["pipe"][3]["embedder"]["load_path"] = "../data/embedder/cc.fr.100.bin"
to_config["chainer"]["pipe"][3]["save_path"] = "./bot/model"
to_config["chainer"]["pipe"][3]["load_path"] = "./bot/model"
to_config["chainer"]["pipe"][3]["nlg_manager"]["template_path"] = "../data/dstc2/dstc2-templates.txt"
to_config["chainer"]["pipe"][3]["tracker"]["slot_names"] = ["pricerange", "this", "category", "type", "heel"]
to_config["chainer"]["pipe"][3]["learning_rate"] = 0.01
to_config["chainer"]["pipe"][3]["hidden_size"] = 128
to_config["chainer"]["pipe"][3]["momentum"] = 0.9
to_config["train"]["batch_size"]= 8

model = train_model(to_config)