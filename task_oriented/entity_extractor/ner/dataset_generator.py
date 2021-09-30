from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
import re


# valid_file = open("C:/Users/user/Documents/GitHub/Intelligent-Chatbot/dstc2_v3/dstc2-val.jsonlist", "rt")

# valid_list = valid_file.read().split("\n")

# print(len(valid_list[0]))

# new_file = open("dstc2-val.jsonlist", "w")

# for line in valid_list:
#     new_file.write(f"{line}\n")

# new_file.close()

data = DSTC2DatasetReader()._read_from_file("./dataset/dstc2-trn.jsonlist")

# ner_valid_file = open("ner_valid.txt", "w", encoding="utf-8")

# for dialog in data:
#     text = dialog[0]["text"]
#     speaker = dialog[1]["text"]

#     if text.strip() != "":
#         for token in text.split(" "):
#             if token in ["mocassin","mocassins", "moccassin", "basket", "baskets", "classique", "classiques", "skecher", "skechers"]:
#                 ner_valid_file.write(f"{token}	B-type\n")
#             elif token in ["modéré", "bon marché", "chère", "raisonable"]:
#                 ner_valid_file.write(f"{token}	B-pricerange\n")
#             elif token in ["femme", "femmes", "meufs", "homme", "hommes", "mecs", "fille", "filles", "fillette", "fillettes", "garçon", "garcon", "garçons", "fils"]:
#                 ner_valid_file.write(f"{token}	B-category\n")
#             elif token in ["long", "demi", "plat", "compensé"]:
#                 ner_valid_file.write(f"{token}	B-heel\n")
#             else:
#                 ner_valid_file.write(f"{token}	O\n")

#         ner_valid_file.write("\n")

#         if not speaker.startswith("api_call"):
#             ner_valid_file.write(f"{speaker}\n")

# ner_valid_file.close()
