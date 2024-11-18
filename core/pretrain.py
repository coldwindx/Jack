import argparse
import os
import re
import sys
import psutil
import pickle
from loguru import logger
import datatable as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  
cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[${os.getpid()}] {cmdline}")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='FastText', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument("--output", "-o", type=str)
args = parser.parse_args()

### load datasets
pattern = r'[\\/=:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*'
train_dataset = dt.fread(args.dataset, fill=True).to_pandas()
train_dataset["channel"] = train_dataset["channel"].apply(lambda x: " ".join(re.split(pattern, x.lower())))

### pretrain model
if args.task == "FastText":
    from gensim.models import FastText
    fasttext = FastText(train_dataset["channel"].tolist(), min_count=5, vector_size=64, word_ngrams=1, workers=32)
    fasttext.save(args.output)
if args.task == "TfidfVectorizer":
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    X = tv.fit_transform(train_dataset["channel"].tolist())
    pickle.dump(tv, open(args.output, "wb"))

logger.info("[+] pretrain.py finished!")