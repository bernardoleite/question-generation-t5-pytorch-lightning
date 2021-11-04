from datasets import load_dataset
from pprint import pprint 
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

import sys
sys.path.append('../')
import random

train_dataset = load_dataset('squad', split='train')
valid_dataset = load_dataset('squad', split='validation')

#Example of context, question and answer
sample_validation_dataset = next(iter(valid_dataset))
pprint (sample_validation_dataset)

#Example of separate context, question and answer
context = sample_validation_dataset['context']
question = sample_validation_dataset['question']
answer = sample_validation_dataset['answers']['text'][0]

#print("\n\n\n")
#print ("context: ",context)
#print ("question: ",question)
#print ("answer: ",answer)

# Create dataframes
pd.set_option("display.max_colwidth", -1)

df_train = pd.DataFrame( columns = ['context', 'answer','question'])
df_validation = pd.DataFrame( columns = ['context', 'answer','question'])

print(df_validation)
print(df_train)

# Fill train dataframe columns 
count_long = 0
count_short = 0

train_indexes = list(range(0, len(train_dataset))) # Create list of train indexes
train_indexes_sample = random.sample(train_indexes, 1000) # Sample n indexes for training

for index,val in enumerate(tqdm(train_dataset)):
    if index in train_indexes_sample: # Select ony 1.000 rows. To delete!!!!!!!!!
        passage = val['context']
        question = val['question']
        answer = val['answers']['text'][0]
        no_of_words = len(answer.split())
        if no_of_words >= 7: # do not consider answers greater (or equal) to 7. Why!!!!!!!!?????
            count_long = count_long + 1
            continue
        else:
            df_train.loc[count_short] = [passage] + [answer] + [question] 
            count_short = count_short + 1       

print ("count_long train dataset: ",count_long)
print ("count_short train dataset: ",count_short)

########################=############################

# Fill dev dataframe columns
count_long = 0
count_short = 0

val_indexes = list(range(0, len(valid_dataset))) # Create list of train indexes
val_indexes_sample = random.sample(val_indexes, 100) # Sample n indexes for training
        
for index,val in enumerate(tqdm(valid_dataset)):
    if index in val_indexes_sample: # Select ony 100 rows. To delete!!!!!!!!!
        passage = val['context']
        question = val['question']
        answer = val['answers']['text'][0]
        no_of_words = len(answer.split())
        if no_of_words >= 7: # do not consider answers greater (or equal) to 7. Why!!!!!!!!?????
            count_long = count_long + 1
            continue
        else:
            df_validation.loc[count_short]= [passage] + [answer] + [question] 
            count_short = count_short + 1       

print ("count_long validation dataset: ",count_long)
print ("count_short validation dataset: ",count_short)

#Shuffle data
df_train = shuffle(df_train)
df_validation = shuffle(df_validation)
#print(df_train.shape)
#print(df_validation.shape)

train_save_path = '../../data/squad_v1_train.csv'
validation_save_path = '../../data/squad_v1_val.csv'
df_train.to_csv(train_save_path, index = False)
df_validation.to_csv(validation_save_path, index = False)