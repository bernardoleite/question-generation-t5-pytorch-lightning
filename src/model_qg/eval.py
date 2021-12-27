from models import T5FineTuner2

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import argparse

def generate(qgmodel: T5FineTuner2, tokenizer: T5Tokenizer,  answer: str, context: str) -> str:
    source_encoding = tokenizer(
        answer,
        context,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=96,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)

def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Generated: ', generated)
    if original_question:
        print('Original : ', original_question)

    print()
    print('Answer: ', answer)
    print('Conext: ', context)
    print('-----------------------------')

def run():
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

    args_dict = dict(
        batch_size= 4,
        max_len_input = 64,
        max_len_output = 96
    )
    args = argparse.Namespace(**args_dict)

    checkpoint_path = "checkpoints/best-checkpoint.ckpt"
    best_model = T5FineTuner2.load_from_checkpoint(checkpoint_path, hparams=args, t5model=t5_model, t5tokenizer=t5_tokenizer)
    best_model.freeze()
    best_model.eval()

    context = "Oxygen is the chemical element with the symbol O and atomic number 8."
    answer = "Oxygen"

    generated = generate(best_model, t5_tokenizer, answer, context)
    show_result(generated, answer, context)


if __name__ == '__main__':
    run()