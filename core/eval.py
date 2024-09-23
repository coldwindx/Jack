import argparse
import os
import sys
from loguru import logger
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import *
from transformers import BertTokenizer
import lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision(precision="high")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', default='/home/zhulin/pretrain/bert_pretrain_uncased/', type=str)
parser.add_argument('--tfidf', default="./common/tfidf.ph",  type=str)
parser.add_argument('--fasttext', default="./common/fasttext.ph",  type=str)
parser.add_argument('--cls_model', default='SingleChannelPredictor', type=str)
parser.add_argument('--cls_dataset', default='SingleChannelDataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--output', default="/mnt/sdd1/data/zhulin/jack/scores/SingleChannelPredictor.npy", type=str)
args = parser.parse_args()

### load dataset
if args.cls_dataset == "SingleChannelDataset":
    from core.dataset import SingleChannelDataset
    tokenizer = BertTokenizer.from_pretrained(args.pretrain, use_fast=True)
    dataset = SingleChannelDataset(args.dataset)
    collate_fn = SingleChannelDataset.collate(tokenizer)
if args.cls_dataset == "DeepRanDataset":
    from core.dataset import DeepRanDataset
    from gensim.models import FastText
    dataset = DeepRanDataset(args.dataset)
    tv = pickle.load(open(args.tfidf, "rb"))
    fasttext = FastText.load(args.fasttext)
    collate_fn = DeepRanDataset.collate(tv, fasttext, np.argmin(tv.idf_))

# dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8)

### load model
if args.cls_model == "SingleChannelPredictor":
    from core.predictor import SingleChannelPredictor
    ckpt = torch.load(args.model)
    predictor = SingleChannelPredictor(**ckpt["hyper_parameters"])
    predictor.load_state_dict(ckpt["state_dict"])
if args.cls_model == "DeepRanPredictor":
    from core.predictor import DeepRanPredictor
    ckpt = torch.load(args.model)
    predictor = DeepRanPredictor(**ckpt["hyper_parameters"])
    predictor.load_state_dict(ckpt["state_dict"])
### start to eval
predictor = predictor.cuda().eval()
trainer = pl.Trainer(enable_checkpointing=False, logger=False, accelerator="cuda", devices=1)
scores = trainer.predict(predictor, dataloaders=dataloader)
scores = torch.cat(scores, dim=0).to("cuda:0")
np.save(open(args.output, "wb+"), scores.cpu().numpy())

### compute metrics
labels = torch.tensor(dataset.labels, device="cuda:0", dtype=torch.int32)
cm = binary_confusion_matrix(scores, labels)
tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

accuracy = binary_accuracy(scores, labels, threshold = 0.5)
precision = binary_precision(scores, labels, threshold = 0.5).item()
recall = binary_recall(scores, labels, threshold = 0.5).item()
f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
auc = binary_auroc(scores, labels).item()
logger.info(f"\ntp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")

logger.info("[+] eval.py execute finished!")