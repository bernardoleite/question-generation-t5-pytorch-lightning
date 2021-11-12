import argparse
import glob
import os
import json
import time
import logging
import random
import re
import sys
from itertools import chain
from string import punctuation
from pprint import pprint
from tqdm import tqdm

import copy
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

# need this because of the following error:
# forrtl: error (200): program aborting due to control-C event
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# Prepare Dataset Structure
class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len_inp=64,max_len_out=96):
        self.path = filepath

        self.passage_column = "context"
        self.answer = "answer"
        self.question = "question"

        self.data = pd.read_csv(self.path)
        #self.data = pd.read_csv(self.path,nrows=1000)

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount =0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            passage,answer,target = self.data.loc[idx, self.passage_column],self.data.loc[idx, self.answer], self.data.loc[idx, self.question]

            input_ = "context: %s  answer: %s </s>" % (passage, answer)
            target = "question: %s </s>" % (str(target))


            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_],
                truncation = True,
                max_length=self.max_len_input, 
                padding='max_length', 
                return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], 
                truncation = True,
                max_length=self.max_len_output, 
                padding='max_length',
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# T5 Finetuner
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, t5model, t5tokenizer, train_dataset, validation_dataset):
        super(T5FineTuner, self).__init__()
        #self.hparams = hparams #https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
        self.save_hyperparameters(hparams)
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset


    def forward( self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
         
         return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,num_workers=4) # why 4?

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size,num_workers=4) # why 4?

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


def run():
    #torch.multiprocessing.freeze_support()
    pl.seed_everything(42)

    train_file_path = '../../data/squad_v1_train.csv'
    validation_file_path = '../../data/squad_v1_val.csv'

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

    print("Repete--------------------->\n")

    train_dataset = QuestionGenerationDataset(t5_tokenizer, train_file_path)
    validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_file_path)

    # Training...
    args_dict = dict(
        batch_size=4
    )

    args = argparse.Namespace(**args_dict)

    model = T5FineTuner(args, t5_model, t5_tokenizer, train_dataset, validation_dataset)

    trainer = pl.Trainer(max_epochs = 3, gpus=1) #progress_bar_refresh_rate=30

    trainer.fit(model)

    print ("Saving model")
    save_path_model = '../../model/'
    save_path_tokenizer = '../../tokenizer/'
    model.model.save_pretrained(save_path_model)
    t5_tokenizer.save_pretrained(save_path_tokenizer)


if __name__ == '__main__':
    run()