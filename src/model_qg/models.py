import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import torch
from torch import nn, optim
import sys

# T5 Finetuner
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, t5model, t5tokenizer):
        super(T5FineTuner, self).__init__()
        #self.hparams = hparams #https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
        self.args = hparams
        self.save_hyperparameters(hparams)
        self.model = t5model
        self.tokenizer = t5tokenizer

    # you might get even better performance by passing decoder_attention_mask when training your model from https://youtu.be/r6XY80Z9eSA
    def forward( self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #decoder_input_ids = decoder_input_ids,
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
        # For decoding during training
        #result = torch.argmax(outputs.logits, dim=-1)
        #decoded_inputs = self.tokenizer.decode(result[0].flatten(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        #print(decoded_inputs)

        loss = outputs[0]
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
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
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.epsilon)
        else:
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.epsilon)
        return optimizer

    # for experimental purposes
"""     def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.epsilon)
            scheduler = {
                'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00001, steps_per_epoch=2698, epochs=20, pct_start=0.3),
                'interval': 'step',
            }
        else:
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.epsilon)
            scheduler = {
                'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00001, steps_per_epoch=2698, epochs=20, pct_start=0.3),
                'interval': 'step',
            }
        return [optimizer], [scheduler] """