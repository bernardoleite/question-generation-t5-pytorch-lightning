import pandas as pd

import json
import sys
sys.path.append('../')

 # Opening JSON file
with open('../../data/du_2017_split/raw/train.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/du_2017_split/raw/dev.json') as dev_json_file:
    validation_data = json.load(dev_json_file)

with open('../../data/du_2017_split/raw/test.json') as test_json_file:
    test_data = json.load(test_json_file)

train_all_compiled = []
for document in train_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            train_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
df_train = pd.DataFrame(train_all_compiled, columns = ['context', 'question','answer'])


val_all_compiled = []
for document in validation_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            val_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
df_validation = pd.DataFrame(val_all_compiled, columns = ['context', 'question','answer'])


test_all_compiled = []
for document in test_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            test_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
df_test = pd.DataFrame(test_all_compiled, columns = ['context', 'question','answer'])

print(len(df_train))
print(len(df_validation))
print(len(df_test))

print("\n")
print(df_train['context'].iloc[75720])
print("\n")
print(df_train['question'].iloc[75720])
print("\n")
print(df_train['answer'].iloc[75720])

print("\n")
print(df_train['context'].iloc[75719])
print("\n")
print(df_train['question'].iloc[75719])
print("\n")
print(df_train['answer'].iloc[75719])

