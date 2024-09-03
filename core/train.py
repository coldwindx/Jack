import argparse
import os
import sys
from loguru import logger
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

from core.sampler import ImbalancedDatasetSampler

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision(precision="high")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', default='/home/zhulin/pretrain/bert_pretrain_uncased/', type=str)
parser.add_argument('--cls_model', default='SingleChannelPredictor', type=str)
parser.add_argument('--cls_dataset', default='SingleChannelDataset', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_epoches', default=30, type=int)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()

### load pretrain model
tokenizer = BertTokenizer.from_pretrained(args.pretrain, use_fast=True)

### load dataset
if args.cls_dataset == "SingleChannelDataset":
    from core.dataset import SingleChannelDataset
    dataset = SingleChannelDataset(args.dataset)
    collate_fn = SingleChannelDataset.collate(tokenizer)

### split dataset
train_size = len(dataset) - 1024 * 64 * 2
train_dataset, valid_dataset, _ = random_split(dataset, [train_size, 1024 * 64, 1024 * 64])
sampler = ImbalancedDatasetSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

### create a model without training
if args.cls_model == "SingleChannelPredictor":
    from core.predictor import SingleChannelPredictor
    predictor = SingleChannelPredictor(
        vocab_size = 30522,
        input_dim=64,
        model_dim=64,
        num_heads=8,
        num_classes=1,
        num_layers=1,
        dropout=0.1,
        input_dropout=0.1,
        lr=1e-4,
        warmup=8,
        max_iters=args.max_epochs * 4,
        weight_decay=1e-6
    )

### crate a trainer
trainer = pl.Trainer(
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), ModelCheckpoint(every_n_epochs=1, save_top_k=-1)],
    # accelerator="gpu", devices=1, num_nodes=1, strategy="ddp",
    accelerator="auto",
    max_epochs=args.max_epoches,
    accumulate_grad_batches=8
)
trainer.logger._default_hp_metric = None

### start to train
trainer.fit(predictor, train_loader, valid_loader)

logger.info("[+] train.py execute finished!")