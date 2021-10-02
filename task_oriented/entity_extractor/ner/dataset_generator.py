from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
import re

types = ["mocassin","mocassins", "moccassin", "basket", "baskets", "classique", "classiques", "skecher", "skechers"]
priceranges = ["modéré", "bon marché", "chère", "raisonable"]
categories = ["femme", "femmes", "meufs", "homme", "hommes", "mecs", "fille", "filles", "fillette", "fillettes", "garçon", "garcon", "garçons", "fils"]
heels = ["long", "demi", "plat", "compensé"]

def conll_format(reference, target):

    data = DSTC2DatasetReader()._read_from_file(reference)

    ner_valid_file = open(target, "w", encoding="utf-8")

    for dialog in data:
        text = dialog[0]["text"]
        speaker = dialog[1]["text"]

        if text.strip() != "":
            for token in text.split(" "):
                if token in types:
                    ner_valid_file.write(f"{token}	B-type\n")
                elif token in priceranges:
                    ner_valid_file.write(f"{token}	B-pricerange\n")
                elif token in categories:
                    ner_valid_file.write(f"{token}	B-category\n")
                elif token in heels:
                    ner_valid_file.write(f"{token}	B-heel\n")
                else:
                    ner_valid_file.write(f"{token}	O\n")

            ner_valid_file.write("\n")

            if not speaker.startswith("api_call"):
                ner_valid_file.write(f"{speaker}\n")

    ner_valid_file.close()

conll_format("../../../data/dstc2/dstc2-trn.jsonlist", "../../../data/dstc2/train.txt")
conll_format("../../../data/dstc2/dstc2-tst.jsonlist", "../../../data/dstc2/test.txt")
conll_format("../../../data/dstc2/dstc2-val.jsonlist", "../../../data/dstc2/valid.txt")
