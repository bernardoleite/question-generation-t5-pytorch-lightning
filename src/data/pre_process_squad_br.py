## SCRIPT BASED ON
# https://colab.research.google.com/drive/18ueLdi_V321Gz37x4gHq8mb4XZSGWfZx?usp=sharing#scrollTo=NkqM9gzvzRrE
# See "Loading the dataset"

import json
import sys
sys.path.append('../')

squad_pt_train_name = 'squad-train-v1.1.json'
squad_pt_val_name = 'squad-dev-v1.1.json'

squad_pt_train_path = '../../data/squad_br_v2/squad-train-v1.1.json'
squad_pt_val_path = '../../data/squad_br_v2/squad-dev-v1.1.json'

files = [
    {"name": squad_pt_train_name, "path": squad_pt_train_path},
    {"name": squad_pt_val_name, "path": squad_pt_val_path}
    ]

for file in files:

    #contar_treino = 0
    
    # Opening JSON file & returns JSON object as a dictionary 
    f = open(file["path"], encoding="utf-8") 
    data = json.load(f) 
    
    # Iterating through the json list 
    entry_list = list()
    id_list = list()

    for row in data['data']: 
        title = row['title']
        
        for paragraph in row['paragraphs']:
            context = paragraph['context']

            #contar_treino = contar_treino + len(paragraph['qas'])

            for qa in paragraph['qas']:
                entry = {}

                qa_id = qa['id']
                question = qa['question']
                answers = qa['answers']
                
                entry['id'] = qa_id
                entry['title'] = title.strip()
                entry['context'] = context.strip()
                entry['question'] = question.strip()
                
                answer_starts = [answer["answer_start"] for answer in answers]
                answer_texts = [answer["text"].strip() for answer in answers]
                entry['answers'] = {}
                entry['answers']['answer_start'] = answer_starts
                entry['answers']['text'] = answer_texts 

                entry_list.append(entry)

        #print("Título: %s | nr questões: %s" % (title, str(contar_treino)))
        #contar_treino = 0

    reverse_entry_list = entry_list[::-1]

    # for entries with same id, keep only last one (corrected texts by the group Deep Learning Brasil)
    unique_ids_list = list()
    unique_entry_list = list()
    for entry in reverse_entry_list:
        qa_id = entry['id']
        if qa_id not in unique_ids_list:
            unique_ids_list.append(qa_id)
            unique_entry_list.append(entry)
        
    # Closing file 
    f.close() 

    new_dict = {}
    new_dict['data'] = unique_entry_list

    file_name = 'processed-' + str(file["name"])
    file_path = '../../data/squad_br_v2/' + file_name

    with open(file_path, 'w') as json_file:
        json.dump(new_dict, json_file)