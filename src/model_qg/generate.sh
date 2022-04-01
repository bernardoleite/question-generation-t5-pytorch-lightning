#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	python train.py \
	 --checkpoint_path "../../checkpoints/2022-04-01_18-26-57/model-epoch=02-val_loss=2.16.ckpt" \
	 --test_df_path "../../data/squad_br/dataframe/df_test_br.pkl" \
	 --model_name "t5-base" \
	 --tokenizer_name "t5-base" \
	 --batch_size 32 \
	 --max_len_input 512 \
	 --max_len_output 96 \
	 --num_beams 5 \
	 --num_return_sequences 1 \
	 --repetition_penalty 1 \
	 --length_penalty 1 \
	 --seed_value ${i}
done