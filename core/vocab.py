import argparse
import os
import re
import sys
import psutil
import joblib
from loguru import logger
import datatable as dt
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  
cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[${os.getpid()}] {cmdline}")

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="/mnt/sdd1/data/zhulin/jack/cdatasets.test.5.csv", type=str)
parser.add_argument("--output", "-o", default="vocab.txt", type=str)
args = parser.parse_args()

### load datasets
dataset = dt.fread(args.dataset, fill=True)
seqs = dataset[:, dt.f.channel].to_list()[0]

### clear dataset
@joblib.delayed
def task(seq):
    pattern = r'[\\/=_:,.;`<>?\^~%\*\'+$!&@\s{}\[\]()]\s*'
    md5_pattern = re.compile(r'\b[a-fA-F0-9]{32}\b') 
    numeric_pattern = re.compile(r'^[#\d]+$')
    words = re.split(pattern, seq.lower())
    # ans = [w for w in words if not md5_pattern.match(w) and not bool(re.search(r'\d', w)) and not 20 < len(w)]
    ans = [w for w in words if not md5_pattern.match(w) and not numeric_pattern.match(w) and not 20 < len(w)]
    return " ".join(ans)

parallel = joblib.Parallel(n_jobs=20)
results = parallel([task(seq) for seq in seqs])

### train a tf-idf model
tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, max_features=30522)
tv.fit(results)
### 显示 TF-IDF重要度排序
tfidf = {k: tv.idf_[v] for k, v in tv.vocabulary_.items()}
tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
print(tfidf[:100])

logger.info("[+] vocab.py finished!")