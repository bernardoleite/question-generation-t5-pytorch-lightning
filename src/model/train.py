import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from pprint import pprint

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

pl.seed_everything(42)

train_file_path = '../../data/squad_v1_train.csv'
validation_file_path = '../../data/squad_v1_val.csv'

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Exploring Tokenizer

# Exploring vocabulary
#print (t5_tokenizer.get_vocab())
#print (len(t5_tokenizer.get_vocab().keys()))

# Example encoding with t5-base tokenizer
sample_encoding = t5_tokenizer.encode_plus("O açame é do cão.",
                                        max_length=64,
                                        pad_to_max_length=True,
                                        truncation=True,
                                        return_tensors="pt")

#print(sample_encoding.keys())
#pprint(sample_encoding)

#print (sample_encoding['input_ids'].shape)
#print (sample_encoding['input_ids'].squeeze().shape)
#print (sample_encoding['input_ids'])

# Sentencepiece tokenizer used by T5
# In sentencepiece when joining to get back a sentence replace _ by space.
tokenized_output = t5_tokenizer.convert_ids_to_tokens(sample_encoding['input_ids'].squeeze())
print(tokenized_output)

decoded_output = t5_tokenizer.decode(sample_encoding['input_ids'].squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
print (decoded_output)
