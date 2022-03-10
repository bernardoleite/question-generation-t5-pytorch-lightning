import pandas as pd
from datasets import load_dataset
import json
import sys
sys.path.append('../')

doclist_train_path = '../../data/du_2017_split/doclist-train.txt'
doclist_validation_path = '../../data/du_2017_split/doclist-dev.txt'
doclist_test_path = '../../data/du_2017_split/doclist-test.txt'

squad_br_train_path = '../../data/squad_br/processed-squad-train-v1.1.json'
squad_br_val_path = '../../data/squad_br/processed-squad-dev-v1.1.json'

#https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
with open(doclist_train_path) as file:
    doclist_train = [line.rstrip() for line in file]
with open(doclist_validation_path) as file:
    doclist_validation = [line.rstrip() for line in file]
with open(doclist_test_path) as file:
    doclist_test = [line.rstrip() for line in file]

# following tutorial...
datasets = load_dataset('json', 
                        data_files={'train': squad_br_train_path, 'validation': squad_br_val_path}, 
                        field='data')

train_all_compiled = []
val_all_compiled = []
test_all_compiled = []

# Get train and get test according to Du (2017) split
for elem in datasets["train"]:
    title = elem["title"]
    context = elem["context"]
    question = elem["question"]
    answer = elem["answers"]["text"][0]
    id = elem["id"]

    if title in doclist_train:
        train_all_compiled.append([title, context, question, answer, id])
    elif title in doclist_test:
        test_all_compiled.append([title, context, question, answer, id])
    elif title in doclist_train and title in doclist_test:
        print("error!")
        print("Title cannot be repeated in sets: ", title)
        sys.exit()
    else:
        print("error!")
        print("Title not found: ", title)
        sys.exit()

# Save to dataframes
train_df = pd.DataFrame(train_all_compiled, columns = ['title','context', 'question', 'answer', 'id'])
test_df = pd.DataFrame(test_all_compiled, columns = ['title','context', 'question', 'answer', 'id'])
print("Train Dataframe completed.")
print("Test Dataframe completed.")

# Get validation data
for elem in datasets["validation"]:
    title = elem["title"]
    context = elem["context"]
    question = elem["question"]
    answer = elem["answers"]["text"][0]
    id = elem["id"]

    if title in doclist_validation:
        val_all_compiled.append([title, context, question, answer, id])
    elif title in doclist_train:
        print("error!")
        print("Title cannot be repeated in sets: ", title)
        sys.exit()
    else:
        print("error!")
        print("Title not found: ", title)
        sys.exit()
validation_df = pd.DataFrame(val_all_compiled, columns = ['title','context', 'question', 'answer', 'id'])

print("Validation Dataframe completed.")
print("\n")
print("Number of train QA-Paragrah pairs: ", len(train_df))
print("Number of validation QA-Paragrah pairs: ", len(validation_df))
print("Number of test QA-Paragrah pairs: ", len(test_df))

train_df.to_pickle("../../data/squad_br/dataframe/df_train_br.pkl")
validation_df.to_pickle("../../data/squad_br/dataframe/df_validation_br.pkl")
test_df.to_pickle("../../data/squad_br/dataframe/df_test_br.pkl")

print("\n","Pickles were generated from dataframes.")