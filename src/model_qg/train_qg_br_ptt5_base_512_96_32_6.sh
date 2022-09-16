#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	taskset --cpu-list 1-30 python train.py \
	 --dir_model_name "qg_br_v2_ptt5_base_512_96_32_6_seed_${i}" \
	 --model_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	 --tokenizer_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	 --train_df_path "../../data/squad_br_v2/dataframe/df_train_br.pkl" \
	 --validation_df_path "../../data/squad_br_v2/dataframe/df_validation_br.pkl" \
	 --test_df_path "../../data/squad_br_v2/dataframe/df_test_br.pkl" \
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