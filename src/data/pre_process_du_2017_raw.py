import pandas as pd

import json
import sys
sys.path.append('../')

 # Opening JSON file
with open('../../data/du_2017_split/raw/json/train.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/du_2017_split/raw/json/dev.json') as dev_json_file:
    validation_data = json.load(dev_json_file)

with open('../../data/du_2017_split/raw/json/test.json') as test_json_file:
    test_data = json.load(test_json_file)

train_all_compiled = []
for document in train_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            train_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
train_df = pd.DataFrame(train_all_compiled, columns = ['context', 'question', 'answer'])

print("Train Dataframe completed.")

val_all_compiled = []
for document in validation_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            val_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
validation_df = pd.DataFrame(val_all_compiled, columns = ['context', 'question', 'answer'])

print("Validation Dataframe completed.")

test_all_compiled = []
for document in test_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            test_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
test_df = pd.DataFrame(test_all_compiled, columns = ['context', 'question', 'answer'])

print("Test Dataframe completed.")
print("\n")
print("Number of train QA-Paragrah pairs: ", len(train_df))
print("Number of validation QA-Paragrah pairs: ", len(validation_df))
print("Number of test QA-Paragrah pairs: ", len(test_df))

train_df.to_pickle("../../data/du_2017_split/raw/dataframe/train_df.pkl")
validation_df.to_pickle("../../data/du_2017_split/raw/dataframe/validation_df.pkl")
test_df.to_pickle("../../data/du_2017_split/raw/dataframe/test_df.pkl")
print("\n","Pickles were generated from dataframes.")

#print("\n")
#print(train_df['context'].iloc[75720])
#print("\n")
#print(train_df['question'].iloc[75720])
#print("\n")
#print(train_df['answer'].iloc[75720])


