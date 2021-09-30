from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import EncoderDecoderModel
import classifier as classif
import torch
from transformers import CamembertTokenizer
import os

CONTEXTS_PATH = "../data/cqa_contexts/"
CONTEXTS_LIST = []
BERT_MODEL_PATH = "../models/camembert"
CLASSIFIER_CHECKPOINT_PATH = "../classifier/checkpoint"
FAQ_CONFIG_PATH = "../QA/FAQ/config.json"
CQA_MODEL_PATH = '../QA/CQA/model/'
OPEN_DOMAIN_MODEL_1 = "../open_domain/empathetic/"
OPEN_DOMAIN_MODEL_2 = "../open_domain/chitchat/"
OD_CLASSIFIER_CONFIG = "../open_domain/classifier/config.json"
TASK_ORIENTED_CONFIG = "../task_oriented/config.json"

NO_ANSWER_MSG = "Désolé je n'ai pas bien compris ce que vous voulez, voulez-vous que je vous associe à un assistant humain ?"

GO_BOT_BYE_MESSAGE = "Vous êtes les bienvenus!"

def get_all_contexts(path):

    files = os.listdir(path)
    temp = []
    contexts = []
    
    for filename in files:
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            temp.append(f)
    
    for file in temp:
        f = open(file, "r").read()
        contexts.append(f)

    return contexts

def get_templates():
    
    templates_list = open("../data/dstc2/dstc2-templates.txt", "r", encoding="utf-8").read().split("\n")
    templates_dict = {}

    for template in templates_list:
        template = template.split("	")
        template_name = template[0]
        template_msg = template[1]
        templates_dict.update({
            template_name : template_msg
        })

    return templates_dict

def get_response(response_dict):

    for key, val in response_dict.items():
        action = key
        slots = val

    if action == "bye":
        to_model.reset()

    message = templates[action]

    for key,val in slots.items():
        if f"#{key}" in message:
            message = message.replace(f"#{key}", val)

    return message

templates = get_templates()

def get_faq_reply(utter): #get FAQ module reply

    max_pred = max(faq_model([utter])[1][0])
    result = faq_model([utter])[0][0]
    print(f"FAQ => {result} : {max_pred}")

    if max_pred > 0.10:
        return  result
    else:
        return ""

def get_cqa_result(question):

    best_score = 0
    best_result = ""
    for context in CONTEXTS_LIST:
        result = cqa_model({'question': question,'context': context})
        if result["score"] > best_score:
            best_score = result["score"]
            best_result = result["answer"]
    print(f"CQA => {best_result}, {best_score}")
    return [best_result, best_score]

def get_cqa_reply(utter): # get CQA module reply

    result = get_cqa_result(utter)

    if result[1] > 0.50:
        return result[0]
    else:
        return ""

def get_QA_reply(utter)-> str: # get QA system reply

    faq_result = get_faq_reply(utter)
    cqa_result = get_cqa_reply(utter)

    if faq_result != "":
        return faq_result
    elif cqa_result != "":
        return cqa_result
    else:
        return NO_ANSWER_MSG

def get_open_domain_reply(utter):  # get OPEN DOMAIN system reply

    if odc_model(["utter"])[0][0] == "chitchat" : 
        input_ids = chitchat_tokenizer(utter, return_tensors="pt").input_ids
        outputs = chitchat_model.generate(input_ids)
        decoded = chitchat_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded

    if odc_model(["utter"])[0][0] == "empathetic" : 
        input_ids = emp_tokenizer(utter, return_tensors="pt").input_ids
        outputs = emp_model.generate(input_ids)
        decoded = emp_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded

def get_task_oriented_reply(utter):

    result = to_model([utter])[0]

    if len(result) == 1: 
        return result[0]
    else:
        result[1] += " API CALL"
        return result[1]

print(">> LOADING CLASSIFIER.........")

tokenizer = CamembertTokenizer.from_pretrained(BERT_MODEL_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = classif.BERTClass()
model.to(device)
classifier = classif.load_model(CLASSIFIER_CHECKPOINT_PATH, model)

print(">> CLASSIFIER WAS LOADED SUCCESSFULY.........")

print(">> LOADING FAQ MODULE...")

faq_config = read_json(FAQ_CONFIG_PATH)
faq_model = build_model(faq_config)

print(">> FAQ MODULE WAS LOADED SUCCESSFULY.........")

print(">> LOADING CQA MODULE...")

CONTEXTS_LIST = get_all_contexts(CONTEXTS_PATH)

cqa_model = pipeline('question-answering', model=CQA_MODEL_PATH, tokenizer=CQA_MODEL_PATH)

print(">> FINISHED LOADING CQA MODULE...")

print(">> LOADING OPEN DOMAIN SYSTEM...")

emp_tokenizer = AutoTokenizer.from_pretrained(OPEN_DOMAIN_MODEL_1)
emp_model = EncoderDecoderModel.from_pretrained(OPEN_DOMAIN_MODEL_1)

chitchat_tokenizer = AutoTokenizer.from_pretrained(OPEN_DOMAIN_MODEL_2)
chitchat_model = EncoderDecoderModel.from_pretrained(OPEN_DOMAIN_MODEL_2)

odc_config = read_json(OD_CLASSIFIER_CONFIG)
odc_model = build_model(odc_config)

print(">> FINISHED OPEN DOMAIN SYSTEM...")

print("LOADING TASK ORIENTED SYSTEM...")

to_config = read_json(TASK_ORIENTED_CONFIG)
to_model = build_model(to_config)

def classify(utter): # classify user utterance (QA/task/chitchat)
    return classif.predict(classifier, utter, device, tokenizer)

def get_reply(utter): # get reply

    selected_class = classify(utter)

    if selected_class == "QA":
        return get_QA_reply(utter)
    elif selected_class == "task":
        return get_task_oriented_reply(utter)
    else:
        return get_open_domain_reply(utter)