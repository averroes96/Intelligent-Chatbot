from __future__ import absolute_import, division, print_function
from deeppavlov.core.common.file import read_json
import csv
import datasets

configs = read_json("configs.json")

DATA_DESCRIPTION = configs["DATA_DESCRIPTION"]
DATA_NAME = configs["DATA_NAME"]
DATA_PATH = configs["DATA_PATH"]

class CustomDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=DATA_NAME,
            version=datasets.Version("1.0.0"),
            description="Full training set",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=DATA_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/averroes96/Intelligent-Chatbot/data/open_domain/",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": DATA_PATH},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(DATA_PATH,'r',encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for i , row in enumerate(csv_reader):
                if i==0 or len(row) == 0:
                    continue
                context = row[0]
                response = row[1]
                yield i-1, {
                    "context": context,
                    "response": response
                }