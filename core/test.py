import argparse
import os
import pickle
import sys
import numpy as np
import psutil
from loguru import logger
from gensim.models import FastText
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from core.dataset import DeepRanDataset

cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[${os.getpid()}] {cmdline}")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', default='/home/zhulin/pretrain/bert_pretrain_uncased/', type=str)
parser.add_argument('--cls_model', default='SingleChannelPredictor', type=str)
parser.add_argument('--cls_dataset', default='SingleChannelDataset', type=str)
parser.add_argument('--dataset', default="/home/zhulin/datasets/cdatasets.test.5.csv",  type=str)
parser.add_argument('--tfidf', default="./common/tfidf.ph",  type=str)
parser.add_argument('--fasttext', default="./common/fasttext.ph",  type=str)
parser.add_argument('--max_epoches', default=30, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument("--output", "-o", type=str)
args = parser.parse_args()


with open(args.tfidf, "rb") as f:
    tv = pickle.load(f)
fasttext = FastText.load(args.fasttext)

dataset = DeepRanDataset(args.dataset)
loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DeepRanDataset.collate(tv, fasttext, np.min(tv.idf_)))


for batch in loader:
    import pdb
    pdb.set_trace()
    pass