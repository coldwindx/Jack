import os
import sys
import datatable as dt
import pandas as pd
import torch
from torch.utils.data import Dataset

### abspath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

class SingleChannelDataset(Dataset):
    def __init__(self, path):
        data = dt.fread(path, fill=True)
        self.seqs = data[:,dt.f.channel].to_numpy().flatten()
        self.labels = data[:,dt.f.label].to_numpy().flatten()
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]
    
    @staticmethod
    def collate(tokenizer):
        def collate_fn(batch):
            df = pd.DataFrame(batch, columns=["seq", "label"])
            padded_sent_seq = tokenizer(df["seq"].to_list(), padding=True, truncation=True, max_length=2048, return_tensors="pt")
            data_length = torch.tensor([sum(mask) for mask in padded_sent_seq["attention_mask"]])
            return padded_sent_seq["input_ids"], padded_sent_seq["attention_mask"], data_length, torch.tensor(df["label"])
        return collate_fn
    
class MultipleChannelDataset(Dataset):
    def __init__(self, path):
        data = dt.fread(path, fill=True)
        self.pseqs = data[:,dt.f.pchannel].to_numpy().flatten()
        self.fseqs = data[:,dt.f.fchannel].to_numpy().flatten()
        self.rseqs = data[:,dt.f.rchannel].to_numpy().flatten()
        self.aseqs = data[:,dt.f.achannel].to_numpy().flatten()
        self.labels = data[:,dt.f.label].to_numpy().flatten()

    def __len__(self):
        return len(self.labels)
    
    def collate(self, tokenizer):
        def collate_fn(batch):
            df = pd.DataFrame(batch, columns=["pseqs", "fseqs", "rseqs", "aseqs", "label"])
            padded_sent_pseq = tokenizer(df["pseq"].to_list(), padding=True, truncation=True, max_length=2048, return_tensors="pt")
            padded_sent_fseq = tokenizer(df["fseqs"].to_list(), padding=True, truncation=True, max_length=2048, return_tensors="pt")
            padded_sent_rseq = tokenizer(df["rseqs"].to_list(), padding=True, truncation=True, max_length=2048, return_tensors="pt")
            padded_sent_aseq = tokenizer(df["aseqs"].to_list(), padding=True, truncation=True, max_length=2048, return_tensors="pt")

            o_input_ids = list(zip(padded_sent_pseq["input_ids"], padded_sent_fseq["input_ids"], padded_sent_rseq["input_ids"], padded_sent_aseq["input_ids"]))
            o_attention_mask = list(zip(padded_sent_pseq["attention_mask"], padded_sent_fseq["attention_mask"], padded_sent_rseq["attention_mask"], padded_sent_aseq["attention_mask"]))
            o_seq_length = [[sum(m0), sum(m1), sum(m2), sum(m3)] for m0, m1, m2, m3 in o_attention_mask]
            return o_input_ids, o_attention_mask, o_seq_length, torch.tensor(df["label"])
        return collate_fn
    
    def __getitem__(self, idx):
        return self.pseqs[idx], self.fseqs[idx], self.rseqs[idx], self.aseqs[idx], self.labels[idx]
