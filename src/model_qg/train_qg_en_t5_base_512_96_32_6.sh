#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	python train.py \
	 --dir_model_name "qg_en_t5_base_512_96_32_6_seed_${i}" \
	 --model_name "t5-base" \
	 --tokenizer_name "t5-base" \
	 --train_df_path "../../data/squad_en_du_2017/raw/dataframe/df_train_en.pkl" \
	 --validation_df_path "../../data/squad_en_du_2017/raw/dataframe/df_validation_en.pkl" \
	 --test_df_path "../../data/squad_en_du_2017/raw/dataframe/df_test_en.pkl" \
	 --max_len_input 512 \
	 --max_len_output 96 \
	 --batch_size 32 \
	 --max_epochs 6 \
	 --patience 3 \
	 --optimizer "AdamW" \
	 --learning_rate 0.0001 \
	 --epsilon 0.000001 \
	 --num_gpus 1 \
	 --seed_value ${i}
done