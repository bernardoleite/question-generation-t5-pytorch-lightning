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
import json

def generate(args, qgmodel: T5FineTuner, tokenizer: T5Tokenizer,  answer: str, context: str) -> str:

    source_encoding = tokenizer(
        answer,
        context,
        max_length=args.max_len_input,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_return_sequences=args.num_return_sequences, # defaults to 1
        num_beams=args.num_beams, # defaults to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! myabe experiment with 5
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty, # defaults to 1.0, #last value was 2.5
        length_penalty=args.length_penalty, # defaults to 1.0
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

def run(args):

    params_dict = dict(
        batch_size = args.batch_size,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output
    )
    params = argparse.Namespace(**params_dict)

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    checkpoint_path = args.checkpoint_path
    best_model = T5FineTuner.load_from_checkpoint(checkpoint_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    best_model.freeze()
    best_model.eval()

    test_df = pd.read_pickle(args.test_df_path)

    predictions = []
    test_df = test_df.sample(n=20) # to delete !!!!!!!!!!!!!!

    for index, row in test_df.iterrows():
        #generated = generate(args, best_model, t5_tokenizer, row['answer'], row['context'])
        generated = "oi gato" # to delete

        predictions.append(
            {'context': row['context'],
            'gt_question': row['question'],
            'answer': row['answer'],
            'gen_question': generated} # to change !!!!!!!!
        )
        #show_result(generated, row['answer'], row['context'], row['question'])

    print("All predictions are completed.")
    print("Number of predictions (q-a-c triples): ", len(predictions))

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = "../../predictions/" + currentdate()
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # Save json to json file
    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + '/predictions.json', 'w', encoding='utf-8') as file:
        json.dump(predictions, file)

    print("Predictions were saved in ", prediction_json_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    parser.add_argument('-mp','--checkpoint_path', type=str, metavar='', default="checkpoints/best-checkpoint.ckpt", required=True, help='Model checkpoint path.')
    parser.add_argument('-tp','--test_df_path', type=str, metavar='', default="../../data/du_2017_split/raw/dataframe/test_df.pkl", required=False, help='Test dataframe path.')

    parser.add_argument('-bs','--batch_size', type=int, metavar='', default=4, required=True, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=64, required=True, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=True, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=1, required=True, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=True, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args)