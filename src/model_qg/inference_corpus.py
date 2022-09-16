from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
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
import torch

def generate(args, device, qgmodel: T5FineTuner, tokenizer: T5Tokenizer,  answer: str, context: str) -> str:

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

    # Put this in GPU (faster than using cpu)
    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = qgmodel.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
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
        print('Original: ', original_question)

    print()
    print('Answer: ', answer)
    print('Context: ', context)
    print('-----------------------------')

def run(args):
    # Load args (needed for model init) and log json
    params_dict = dict(
        checkpoint_model_path = args.checkpoint_model_path,
        predictions_save_path = args.predictions_save_path,
        test_df_path = args.test_df_path,
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        batch_size = args.batch_size,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        num_beams = args.num_beams,
        num_return_sequences = args.num_return_sequences,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
        seed_value = args.seed_value
    )
    params = argparse.Namespace(**params_dict)

    # Load T5 base Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    # Load T5 base Model
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    # Load T5 fine-tuned model for QG
    checkpoint_model_path = args.checkpoint_model_path
    qgmodel = T5FineTuner.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    # Not sure if this should be after or before changing device for inference.
    qgmodel.freeze()
    qgmodel.eval()

    # Read test data
    test_df = pd.read_pickle(args.test_df_path)
    test_df = test_df.sample(n=20) # to DELETEEEEEE !!!!!!!!!!!!!!

    predictions = []

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference:", device)

    # Generate questions and append predictions
    start_time_generate = time.time()
    printcounter = 0
    for index, row in test_df.iterrows():
        generated = generate(args, device, qgmodel, t5_tokenizer, row['answer'], row['context'])

        predictions.append(
            {'context': row['context'],
            'gt_question': row['question'],
            'answer': row['answer'],
            'gen_question': generated} # to change !!!!!!!! what?
        )
        printcounter += 1
        if (printcounter == 400):
            print(str(printcounter) + " questions have been generated.")
            printcounter = 0
        #show_result(generated, row['answer'], row['context'], row['question'])

    print("All predictions are completed.")
    print("Number of predictions (q-a-c triples): ", len(predictions))

    end_time_generate = time.time()
    gen_total_time = end_time_generate - start_time_generate
    print("Inference time: ", gen_total_time)

    # Save questions and answers to json file

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = args.predictions_save_path
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # Save json to json file
    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + 'predictions.json', 'w', encoding='utf-8') as file:
        json.dump(predictions, file)

    # Save json params to json file next to predictions
    with open(prediction_json_path + 'params.json', 'w', encoding='utf-8') as file:
        file.write(
            '[' +
            ',\n'.join(json.dumps(str(key)+': '  + str(value)) for key,value in params_dict.items()) +
            ']\n')

    print("Predictions were saved in ", prediction_json_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/qg_br_ptt5_base_512_96_32_6_seed_42/model-epoch=03-val_loss=1.68.ckpt", required=False, help='Model folder checkpoint path.')
    parser.add_argument('-psp','--predictions_save_path', type=str, metavar='', default="../../predictions/qg_br_ptt5_base_512_96_32_6_seed_42/model-epoch=03-val_loss=1.68/", required=False, help='Folder path to save predictions after inference.')
    parser.add_argument('-tp','--test_df_path', type=str, metavar='', default="../../data/squad_br_v2/dataframe/df_test_br.pkl", required=False, help='Test dataframe path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="unicamp-dl/ptt5-base-portuguese-vocab", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="unicamp-dl/ptt5-base-portuguese-vocab", required=False, help='Tokenizer name.')

    parser.add_argument('-bs','--batch_size', type=int, metavar='', default=32, required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')
    parser.add_argument('-sv','--seed_value', type=int, default=42, metavar='', required=False, help='Seed value.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args)