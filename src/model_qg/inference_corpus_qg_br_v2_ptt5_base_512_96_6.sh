#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	CUDA_VISIBLE_DEVICES=1 python inference_corpus.py \
	 --checkpoint_model_path "../../checkpoints/qg_br_ptt5_base_512_96_32_6_seed_42/model-epoch=01-val_loss=1.52.ckpt" \
	 --predictions_save_path "../../predictions/qg_br_ptt5_base_512_96_32_6_seed_42/model-epoch=01-val_loss=1.52/" \
	 --test_df_path "../../data/squad_br_v2/dataframe/df_test_br.pkl" \
	 --model_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	 --tokenizer_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	 --batch_size 32 \
	 --max_len_input 512 \
	 --max_len_output 96 \
	 --num_beams 5 \
	 --num_return_sequences 1 \
	 --repetition_penalty 1 \
	 --length_penalty 1 \
	 --seed_value ${i}
done