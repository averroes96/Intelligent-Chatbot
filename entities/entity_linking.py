from pathlib import Path
import csv
import spacy

def load_entities():
    output_dir =Path.cwd().parent / "output"
    entities_loc = output_dir / "entities.csv"

    names = dict()
    descriptions = dict()

    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id = row[0]
            name = row[1]
            desc = row[2]
            names[id] = name
            descriptions[id] = desc
    
    return names, descriptions


if __name__ = "__main__":
    nlp = spacy.load()
