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

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

from transformers import AutoTokenizer
class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    #self.save_hyperparameters()
    self.myparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def is_logger(self):
    return True # self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    # {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    return None

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.myparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.myparams.learning_rate, eps=self.myparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  # handled automatically by PyTorch Lightning
  # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  #   if self.trainer.use_tpu:
  #     xm.optimizer_step(optimizer)
  #   else:
  #     optimizer.step()
  #   optimizer.zero_grad()
  #   self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.myparams)
    dataloader = DataLoader(train_dataset, batch_size=self.myparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.myparams.train_batch_size * max(1, self.myparams.n_gpu)))
        // self.myparams.gradient_accumulation_steps
        * float(self.myparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.myparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.myparams)
    return DataLoader(val_dataset, batch_size=self.myparams.eval_batch_size, num_workers=4)

args_dict = dict(
    output_dir="/home/joaquimneto04/t5_webnlg", # path to save the checkpoints
    model_name_or_path='google/byt5-base',
    tokenizer_name_or_path='google/byt5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')

from torch.utils.data import Dataset, DataLoader
from random import random

class WebNLGDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path, max_len=512):
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []
    self.type_path = type_path

    self._build_examples_from_files()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()
    target_mask = self.targets[index]["attention_mask"].squeeze()

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

  def _build_examples_from_files(self):
      data = pd.read_csv('/home/joaquimneto04/WebNLG/webNLG2020_'+self.type_path+'.csv')
        
      triples = data['input_text'].values.tolist()
      texts = data['target_text'].values.tolist()
      
      for triple in triples:
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [triple], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
        )

        self.inputs.append(tokenized_inputs)

      for txt in texts:
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [txt], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
        )

        self.targets.append(tokenized_targets)

"""### Train"""

args_dict.update({'data_dir': '/home/joaquimneto04/WebNLG', 'output_dir': '/home/joaquimneto04/t5_webnlg', 'num_train_epochs': 1, 'vocab_file': 'tokenizer_config.json'})
args = argparse.Namespace(**args_dict)

# add this in?:
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# callbacks=[EarlyStopping(monitor='val_loss')]

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    #amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
)

def get_dataset(tokenizer, type_path, args):
  return WebNLGDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

model.model.save_pretrained('/home/joaquimneto04/trained_model/t5_base_webnlg')
model.model.eval()
