#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	taskset --cpu-list 1-30 python train.py \
	 --dir_model_name "qg_en_mt5_base_512_96_16_6_seed_${i}" \
	 --model_name "google/mt5-base" \
	 --tokenizer_name "google/mt5-base" \
	 --train_df_path "../../data/squad_en_du_2017/raw/dataframe/df_train_en.pkl" \
	 --validation_df_path "../../data/squad_en_du_2017/raw/dataframe/df_validation_en.pkl" \
	 --test_df_path "../../data/squad_en_du_2017/raw/dataframe/df_test_en.pkl" \
	 --max_len_input 512 \
	 --max_len_output 96 \
	 --batch_size 16 \
	 --max_epochs 6 \
	 --patience 3 \
	 --optimizer "AdamW" \
	 --learning_rate 0.0001 \
	 --epsilon 0.000001 \
	 --num_gpus 1 \
	 --seed_value ${i}
done