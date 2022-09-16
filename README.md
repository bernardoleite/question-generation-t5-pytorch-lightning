# Question Generation with T5 model using 🤗Transformers and Pytorch Lightning
===============

Sample source code and models for our [EPIA 2022](https://epia2022.inesc-id.pt/) paper: [Neural Question Generation for the Portuguese Language: A Preliminary Study](https://link.springer.com/chapter/10.1007/978-3-031-16474-3_63)

> **Abstract:** *Question Generation* (QG) is an important and challenging problem that has attracted attention from the natural language processing (NLP) community over the last years. QG aims to automatically generate questions given an input. Recent studies in this field typically use widely available question-answering (QA) datasets (in English) and neural models to train and build these QG systems. As lower-resourced languages (e.g. Portuguese) lack large-scale quality QA data, it becomes a significant challenge to experiment with recent neural techniques. This study uses a Portuguese machine-translated version of the SQuAD v1.1 dataset to perform a preliminary analysis of a neural approach to the QG task for Portuguese. We frame our approach as a sequence-to-sequence problem by fine-tuning a pre-trained language model – T5 for generating factoid (or wh)-questions. Despite the evident issues that a machine-translated dataset may bring when using it for training neural models, the automatic evaluation of our Portuguese neural QG models presents results in line with those obtained for English. To the best of our knowledge, this is the first study addressing Neural QG for Portuguese.

**Authors:** Bernardo Leite, Henrique Lopes Cardoso

If you use this research in your work, please kindly cite us:
```bibtex
@inproceedings{leite_2022_nqg,
	title        = {Neural Question Generation for the Portuguese Language: A Preliminary Study},
	author       = {Leite, Bernardo and Lopes Cardoso, Henrique},
	year         = 2022,
	booktitle    = {Progress in Artificial Intelligence},
	publisher    = {Springer International Publishing},
	address      = {Cham},
	pages        = {780--793},
	isbn         = {978-3-031-16474-3},
	editor       = {Marreiros, Goreti and Martins, Bruno and Paiva, Ana and Ribeiro, Bernardete and Sardinha, Alberto}
}
```

## Some examples
_Afonso Henriques, também chamado de Afonsinho, e cognominado de "o Conquistador", foi o primeiro Rei de Portugal. Passa a intitular-se "Rei dos Portugueses" a partir de 1140 e reinou de jure a partir de 5 de outubro de 1143, com a celebração do Tratado de Zamora, até à sua morte. Era filho de Henrique, Conde de Portucale e sua esposa Teresa de Leão._

**Answer**: Afonso Henriques
**Generated Question**: Quem foi o primeiro rei de Portugal?

**Answer**: Afonsinho
**Generated Question**: Qual era o outro nome para Afonso Henriques?

**Answer**: "o Conquistador"
**Generated Question**: Qual era o apelido de Afonso Henriques?

**Answer**: 1143
**Generated Question**: Em que ano foi celebrado o Tratado de Zamora?

**Answer**: Conde de Portucale
**Generated Question**: Quem era o pai de Afonso Henriques?

## Main Features
* Training, inference and evaluation scripts for QG
* Fine-tuned QG T5 models for both English and Portuguese

## Prerequisites
```bash
Python 3 (tested with version 3.9.13 on Windows 10)
```

## Installation and Configuration
1. Clone this project:
    ```python
    git clone https://github.com/bernardoleite/question-generation-t5-pytorch-lightning
    ```
2. Install the Python packages from [requirements.txt](https://github.com/bernardoleite/question-generation-t5-pytorch-lightning/blob/main/requirements.txt). If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:
    ```bash
    cd question-generation-t5-pytorch-lightning/
    pip install -r requirements.txt
    ```

## Usage
You can use this code for **data preparation**, **training**, **inference/predicting** (full corpus or individual sample), and **evaluation**.

### Data preparation
Current experiments use the SQuAD v1.1 dataset for English (original) and Portuguese (machine-translated) versions. So the next steps are specifically intended to preparing this dataset, but the same approach is applicable to other data types.
* Example for preparing the **English** (original) SQuAD v1.1 dataset:
1.  Create `squad_en_du_2017` folder inside `data` folder
2.  Download the files and folders from [here](https://github.com/xinyadu/nqg/tree/master/data) and place them inside `data/squad_en_du_2017` folder
2.  Create `dataframe` folder inside `data/squad_en_du_2017/raw`
3.  Go to `src/data`. By running `src/data/pre_process_du_2017_raw.py` the following dataframes (pickle format) will be created inside `data/squad_en_du_2017/raw/dataframe`: `df_train_en.pkl`, `df_validation_en.pkl` and `df_test_en.pkl`.

**Important note for the English version**: Regardless of the data type, make sure the dataframe columns follow this scheme: [**context**, **question**, **answer**].

* Example for preparing the **Portuguese** (machine-translated) SQuAD v1.1 dataset:
1.  Download `squad-train-v1.1.json` and `squad-dev-v1.1.json` data from [here](https://github.com/nunorc/squad-v1.1-pt).
2.  Create `data/squad_br_v2` and copy previous files inside.
3.  Create `data/squad_br_v2/dataframe`
4.  Go to `src/data`. By running `src/data/pre_process_squad_br.py` and then `src/data/pre_process_squad_br_processed.py` the following dataframes (pickle format) will be created inside `data/squad_br_v2/dataframe/`: `df_train_br.pkl`, `df_validation_br.pkl` and `df_test_br.pkl`.

**Important note for the Portuguese version**: Regardless of the data type, make sure the dataframe columns follow this scheme: [**title**, **context**, **question**, **answer**, **id**].

### Training 
1.  Go to `src/model_qg`. The file `train.py` is responsible for the training routine. Type the following command to read the description of the parameters:
    ```bash
    python train.py -h
    ```
    You can also run the example training script (linux and mac) `train_qg_en_t5_base_512_96_32_6.sh`:
    ```bash
    bash train_qg_en_t5_base_512_96_32_6.sh
    ```
    The previous script will start the training routine with predefined parameters:
    ```python
    #!/usr/bin/env bash

	for ((i=42; i <= 42; i++))
	do
		taskset --cpu-list 1-30 python train.py \
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
    ```

2. In the end, all model information is available at `checkpoints/checkpoint-name`. The information includes models checkpoints for each epoch (`*.ckpt` files), tensorboard logs (`tb_logs/`) and csv logs (`csv_logs/`).

3. Previous steps also apply to `train_qg_br_ptt5_base_512_96_32_6.sh` for the training routine in **Portuguese**.


### Inference (full corpus)
1.  Go to `src/model_qg`. The file `inference_corpus.py` is responsible for the inference routine (full corpus) given a certain **model checkpoint**. Type the following command to read the description of the parameters:
    ```bash
    python inference_corpus.py -h
    ```
    You can also run the example inference corpus script (linux and mac) `inference_corpus_qg_en_t5_base_512_96_6.sh`:
    ```bash
    bash inference_corpus_qg_en_t5_base_512_96_6.sh
    ```
    The previous script will start the inference routine with predefined parameters for the model checkpoint `model-epoch=00-val_loss=0.32.ckpt`:
    ```python
    #!/usr/bin/env bash

    for ((i=42; i <= 42; i++))
    do
        CUDA_VISIBLE_DEVICES=1 python inference_corpus.py \
        --checkpoint_model_path "../../checkpoints/qa_en_t5_base_512_96_32_10_seed_42/model-epoch=00-val_loss=0.32.ckpt" \
        --predictions_save_path "../../predictions/qa_en_t5_base_512_96_32_10_seed_42/model-epoch=00-val_loss=0.32/" \
        --test_df_path "../../data/squad_en_original/processed/df_validation_en.pkl" \
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
    ```

2. In the end, predictions will be available at `predictions/checkpoint-name`. The folder contains model predictions (`predictions.json`), and parameters (`params.json`).

3. Previous steps also apply to `inference_corpus_qg_br_v2_ptt5_base_512_96_6.sh` for the inference routine in **Portuguese**.


### Inference (individual sample)
Go to `src/model_qg`. The file `inference_example.py` is responsible for the inference routine (individual sample) given a certain **model checkpoint**, **CONTEXT** and **ANSWER**. Type the following command to read the description of the parameters:

```bash
python inference_example.py -h
```
Example/Demo:

1.  Change **ANSWER** and **CONTEXT** variables in `inference_example.py`:
    ```python
    ANSWER = 'Rei dos Portugueses'
    CONTEXT = """Afonso Henriques, também chamado de Afonsinho, e cognominado de "o Conquistador", foi o primeiro Rei de Portugal. Passa a intitular-se "Rei dos Portugueses" a partir de 1140 e reinou de jure a partir de 5 de outubro de 1143, com a celebração do Tratado de Zamora, até à sua morte. Era filho de Henrique, Conde de Portucale e sua esposa Teresa de Leão, que, à morte do conde Henrique, "ascende rapidamente ao governo do condado, o que confirma o carácter hereditário que o mesmo possuía."""
    ```
2.  Run `inference_example.py` (e.g., using `model-epoch=00-val_loss=0.32.ckpt`).
3.  See output (it should be a question)

### Evaluation 
To do.

### Checkpoints 
To do.

## Acknowledgements
This project is inspired by/based on the implementations of [Venelin Valkov](https://www.youtube.com/watch?v=r6XY80Z9eSA&t=1994s), [Ramsri Golla](https://www.udemy.com/course/question-generation-using-natural-language-processing/), [Suraj Patil](https://github.com/patil-suraj/question_generation) and [Kristiyan Vachev](https://github.com/KristiyanVachev/Question-Generation).
   
## Contact
* Bernardo Leite, bernardo.leite@fe.up.pt
* Henrique Lopes Cardoso, hlc@fe.up.pt