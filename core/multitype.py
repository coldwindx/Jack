import os
import sys
import pandas as pd
import datatable as dt
import numpy as np
import torch
from torcheval.metrics.functional import *

### abspath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

### Ransomeware Sample
virlocks = ["k74d200bcb1fae1e6dabdaa115c051099", "kd5fee0c6f1d0d730de259c64e6373a0c", 
            "kb65b194c6cc134d56ba3acdcc7bd3051", "kb99c2748e46c0f8ed8da08fd933e0d9f"]
mbrs = ["ke3b7d39be5e821b59636d0fe7c2944cc", "kaf2379cc4d607a45ac44d62135fb7015", "keba85b706259f4dc0aec06a6a024609a",
        "k74d9610a72fa9ed105c927e3b1897c5b", "k8c64c2ff302f64cf326897af8176d68e", "k5e271dbfb5803f600b30f7d9945024fd"]

### Read dataset and scores
test_data = dt.fread("/mnt/sdd1/data/zhulin/jack/cdatasets.test.6.csv", fill=True)
scores = np.load(open("/mnt/sdd1/data/zhulin/jack/scores/SingleChannelPredictor.6.npy", "rb"))
test_data["score"] = scores
### Read family
df = pd.read_csv("/mnt/sdd1/data/zhulin/jack/samples.csv")

### Split dataset
result = []
for index in df[df["datasettype"] == "test"]["esindex"]:
    if index in mbrs:
        result.append(test_data[dt.f.index == index, [dt.f.score, dt.f.label]])
ds = dt.rbind(result).to_list()

scores = torch.tensor(ds[0], device="cuda:0", dtype=torch.float32)
labels = torch.tensor(ds[1], device="cuda:0", dtype=torch.int32)

accuracy = binary_accuracy(scores, labels, threshold = 0.5)
precision = binary_precision(scores, labels, threshold = 0.5).item()
recall = binary_recall(scores, labels, threshold = 0.5).item()
f1 = binary_f1_score(scores, labels, threshold = 0.5).item()
auc = binary_auroc(scores, labels).item()
print(f"\nacc: {accuracy}\npre: {precision}\nrec: {recall}\nauc: {auc}\nf1: {f1}")

