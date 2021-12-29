import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

# T5 Finetuner
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, t5model, t5tokenizer): #, train_dataset, validation_dataset, test_dataset
        super(T5FineTuner, self).__init__()
        #self.hparams = hparams #https://github.com/PyTorchLightning/pytorch-lightning/discussions/7525
        self.save_hyperparameters(hparams)
        self.model = t5model
        self.tokenizer = t5tokenizer
        #self.train_dataset = train_dataset
        #self.validation_dataset = validation_dataset
        #self.test_dataset = test_dataset

    # you might get even better performance by passing decoder_attention_mask when training your model from https://youtu.be/r6XY80Z9eSA
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
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer

    #def train_dataloader(self):
        #return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,num_workers=4) # why 4?

    #def val_dataloader(self):
        #return DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size,num_workers=4) # why 4?

    #def test_dataloader(self):
        #return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,num_workers=4) # why 4?