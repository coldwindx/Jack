import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import *
from transformers import BertTokenizer
import lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

from common.extractor import FeatureExtractor

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision(precision="high")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', default='/home/zhulin/pretrain/bert_pretrain_uncased/', type=str)
parser.add_argument('--cls_model', default='SingleChannelPredictor', type=str)
parser.add_argument('--cls_dataset', default='SingleChannelDataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()

### load pretrain model
tokenizer = BertTokenizer.from_pretrained(args.pretrain, use_fast=True)
# padded_sent_seq = tokenizer("rtldestroymemoryblocklookaside", padding=True, truncation=True, max_length=2048, return_tensors="pt")
token = tokenizer.tokenize("ldrsetappcompatdllredirectioncallback")
print(token)