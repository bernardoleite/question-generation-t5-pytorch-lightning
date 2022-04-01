import pandas as pd

import json
import sys
sys.path.append('../')

 # Opening JSON file
with open('../../data/squad_en_original/train-v1.1.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/squad_en_original/dev-v1.1.json') as dev_json_file:
    validation_data = json.load(dev_json_file)

NR_CHAR = 0

train_all_compiled = []
for document in train_data["data"]:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        NR_CHAR = NR_CHAR + len(context)
        qas = para["qas"]
        for qa in qas:
            NR_CHAR = NR_CHAR + len(qa["question"])
            for ans in qa["answers"]:
                NR_CHAR = NR_CHAR + len(ans)
            train_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
train_df = pd.DataFrame(train_all_compiled, columns = ['context', 'question', 'answer'])
print("Nr of total chars for TRAIN ONLY: ", NR_CHAR)


val_all_compiled = []
for document in validation_data["data"]:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        NR_CHAR = NR_CHAR + len(context)
        qas = para["qas"]
        for qa in qas:
            NR_CHAR = NR_CHAR + len(qa["question"])
            for ans in qa["answers"]:
                NR_CHAR = NR_CHAR + len(ans)
            val_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
validation_df = pd.DataFrame(val_all_compiled, columns = ['context', 'question', 'answer'])

print("Nr of totals chars for TAIN and VALIDATION: ", NR_CHAR)
print("Total price for translating with DeepL: ", (NR_CHAR/1000000)*20)