import argparse
import os
import sys
from loguru import logger
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
parser.add_argument('--cls_model', default='SingleChannelPredictor', type=str)
parser.add_argument('--cls_dataset', default='SingleChannelDataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()

### load pretrain model
tokenizer = BertTokenizer.from_pretrained(args.pretrain, use_fast=True)

### load dataset
if args.cls_dataset == "SingleChannelDataset":
    from core.dataset import SingleChannelDataset
    dataset = SingleChannelDataset(args.dataset)
    collate_fn = SingleChannelDataset.collate(tokenizer)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

### load model
if args.cls_model == "SingleChannelPredictor":
    from core.predictor import SingleChannelPredictor
    from core.predictor import SingleChannelPredictor
    ckpt = torch.load(args.model)
    ckpt['state_dict']['net.input_net.0.weight'] = ckpt['state_dict'].pop('input_net.0.weight')
    ckpt['state_dict']['net.input_net.2.weight'] = ckpt['state_dict'].pop('input_net.2.weight')
    ckpt['state_dict']['net.input_net.2.bias'] = ckpt['state_dict'].pop('input_net.2.bias')
    ckpt['state_dict']['net.transformer.layers.0.attn.qkv_proj.weight'] = ckpt['state_dict'].pop('transformer.layers.0.self_attn.qkv_proj.weight')
    ckpt['state_dict']['net.transformer.layers.0.attn.qkv_proj.bias'] = ckpt['state_dict'].pop('transformer.layers.0.self_attn.qkv_proj.bias')
    ckpt['state_dict']['net.transformer.layers.0.attn.o_proj.weight'] = ckpt['state_dict'].pop('transformer.layers.0.self_attn.o_proj.weight')
    ckpt['state_dict']['net.transformer.layers.0.attn.o_proj.bias'] = ckpt['state_dict'].pop('transformer.layers.0.self_attn.o_proj.bias')
    ckpt['state_dict']['net.transformer.layers.0.linear_net.0.weight'] = ckpt['state_dict'].pop('transformer.layers.0.linear_net.0.weight')
    ckpt['state_dict']['net.transformer.layers.0.linear_net.0.bias'] = ckpt['state_dict'].pop('transformer.layers.0.linear_net.0.bias')
    ckpt['state_dict']['net.transformer.layers.0.linear_net.3.weight'] = ckpt['state_dict'].pop('transformer.layers.0.linear_net.3.weight')
    ckpt['state_dict']['net.transformer.layers.0.linear_net.3.bias'] = ckpt['state_dict'].pop('transformer.layers.0.linear_net.3.bias')
    ckpt['state_dict']['net.transformer.layers.0.norm1.weight'] = ckpt['state_dict'].pop('transformer.layers.0.norm1.weight')
    ckpt['state_dict']['net.transformer.layers.0.norm1.bias'] = ckpt['state_dict'].pop('transformer.layers.0.norm1.bias')
    ckpt['state_dict']['net.transformer.layers.0.norm2.weight'] = ckpt['state_dict'].pop('transformer.layers.0.norm2.weight')
    ckpt['state_dict']['net.transformer.layers.0.norm2.bias'] = ckpt['state_dict'].pop('transformer.layers.0.norm2.bias')
    ckpt['state_dict']['net.output_net.0.weight'] = ckpt['state_dict'].pop('output_net.0.weight')
    ckpt['state_dict']['net.output_net.0.bias'] = ckpt['state_dict'].pop('output_net.0.bias')
    ckpt['state_dict']['net.output_net.2.weight'] = ckpt['state_dict'].pop('output_net.2.weight')
    ckpt['state_dict']['net.output_net.2.bias'] = ckpt['state_dict'].pop('output_net.2.bias')
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
        max_iters=30 * 4,
        weight_decay=1e-6
    )
    predictor.load_state_dict(ckpt['state_dict'])

### start to eval
predictor = predictor.eval()
trainer = pl.Trainer(enable_checkpointing=False, logger=False)
scores = trainer.predict(predictor, dataloaders=dataloader)
scores = torch.cat(scores, dim=0)

### compute metrics
labels = torch.tensor(dataset.labels, device="cuda")
cm = binary_confusion_matrix(scores, labels)
tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

accuracy = binary_accuracy(scores, labels, threshold = 0.5)
precision = binary_precision(scores, labels, threshold = 0.5).item()
recall = binary_recall(scores, labels, threshold = 0.5).item()
f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
auc = binary_auroc(scores, labels).item()
logger.info(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")

logger.info("[+] eval.py execute finished!")