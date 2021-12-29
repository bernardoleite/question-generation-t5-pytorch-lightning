from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import pandas as pd
import sys
sys.path.append('../')

from models import T5FineTuner
from utils import currentdate
import time
import os

def generate(qgmodel: T5FineTuner, tokenizer: T5Tokenizer,  answer: str, context: str) -> str:

    source_encoding = tokenizer(
        answer,
        context,
        max_length=64,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_return_sequences=1, # defaults to 1
        num_beams=5, # defaults to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! myabe experiment with 5
        max_length=96,
        repetition_penalty=1.0, # defaults to 1.0, #last value was 2.5
        length_penalty=1.0, # defaults to 1.0
        early_stopping=True, # defaults to False
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    
    return ''.join(preds)

def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Generated: ', generated)
    if original_question:
        print('Original : ', original_question)

    print()
    print('Answer: ', answer)
    print('Context: ', context)
    print('-----------------------------')

def run():
    args_dict = dict(
        batch_size = 4,
        max_len_input = 64,
        max_len_output = 96
    )
    args = argparse.Namespace(**args_dict)

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    checkpoint_path = "checkpoints/best-checkpoint.ckpt"
    best_model = T5FineTuner.load_from_checkpoint(checkpoint_path, hparams=args, t5model=t5_model, t5tokenizer=t5_tokenizer)

    best_model.freeze()
    best_model.eval()

    test_df = pd.read_pickle("../../data/du_2017_split/raw/dataframe/test_df.pkl")

    all_contexts = []
    all_gt_questions = []
    all_answers = []
    all_gen_questions = []

    #test_df = test_df.sample(n=20) # to delete

    for index, row in test_df.iterrows():
        context = row['context']
        if '\n' in context:
            context = context.replace("\n", "") # hard coded fix! To be changed!!!!!!!!!!!!!!!!!!!
        all_contexts.append(context)
        all_gt_questions.append(row['question'])
        all_answers.append(row['answer'])
        #generated = generate(best_model, t5_tokenizer, row['answer'], row['context'])
        all_gen_questions.append("oi gato")

        #show_result(generated, row['answer'], row['context'], row['question'])

    print("All predictions are completed.")
    print(len(all_contexts))
    print(len(all_gt_questions))
    print(len(all_answers))
    print(len(all_gen_questions))

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    folder_path = "../../predictions/" + currentdate()
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    file_name = "all_contexts.txt"
    file_path = folder_path + "/" + file_name

    #https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python/899176
    with open(file_path, 'w', encoding="utf-8") as f:
        for item in all_contexts:
            f.write("%s\n" % item)

    print("Predictions saved in ", file_path)

if __name__ == '__main__':
    run()