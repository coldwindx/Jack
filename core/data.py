### Dataset格式转换，from json to csv
import os
import sys
import argparse
from loguru import logger
import pandas as pd
import psutil
from tqdm import tqdm
import json
import collections
import datatable as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[$] {os.getpid()} {cmdline}")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str)
parser.add_argument("--json", type=str)
parser.add_argument("--csv", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()

def json_to_csv(json, csv):
    ### read json data
    collect = collections.defaultdict(list)
    for file in json.split(","):
        logger.info(f"[+] load {file} ...")
        with open(file, "r") as f:
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                for k, v in data.items():
                    collect[k].append(v)

    ### transform json to csv
    df = dt.Frame(collect)
    df.to_csv(csv)

def add_family(csv):
    df_family = pd.read_csv("common/family.csv")
    df_family["index"] = df_family["esindex"]
    for file in csv.split(","):
        df: pd.DataFrame = dt.fread(file, fill=True).to_pandas()
        df["family"] = pd.merge(df, df_family, on="index")["family"]
        df.to_csv(file, index=False)

def add_total(csv):
    cnt = []
    with open("/home/zhulin/workspace/Sun-agent/build/total.eval.json", "r") as f:
        for line in f.readlines():
            cnt.append(json.loads(line)["cnt_event"])
    df: pd.DataFrame = dt.fread(csv, fill=True).to_pandas()
    df["cnt"] = cnt
    df.to_csv(csv, index=False)


def model_2_model(model):
    import torch
    from core.predictor import SingleChannelPredictor
    ckpt = torch.load(model)

    # ckpt["state_dict"]['net.transformer.layers.0.attn.qkv_proj.weight'] = ckpt["state_dict"]['net.transformer.layers.0.self_attn.qkv_proj.weight']
    # ckpt["state_dict"]['net.transformer.layers.0.attn.qkv_proj.bias'] = ckpt["state_dict"]['net.transformer.layers.0.self_attn.qkv_proj.bias']
    # ckpt["state_dict"]['net.transformer.layers.0.attn.o_proj.weight'] = ckpt["state_dict"]['net.transformer.layers.0.self_attn.o_proj.weight']
    # ckpt["state_dict"]['net.transformer.layers.0.attn.o_proj.bias'] = ckpt["state_dict"]['net.transformer.layers.0.self_attn.o_proj.bias']

    # del ckpt["state_dict"]['net.transformer.layers.0.self_attn.qkv_proj.weight']
    # del ckpt["state_dict"]['net.transformer.layers.0.self_attn.qkv_proj.bias']
    # del ckpt["state_dict"]['net.transformer.layers.0.self_attn.o_proj.weight']
    # del ckpt["state_dict"]['net.transformer.layers.0.self_attn.o_proj.bias']

    ckpt["state_dict"]['net.input_net.0.weight'] = ckpt["state_dict"]['input_net.0.weight']
    ckpt["state_dict"]['net.input_net.2.weight'] = ckpt["state_dict"]['input_net.2.weight']
    ckpt["state_dict"]['net.input_net.2.bias'] = ckpt["state_dict"]['input_net.2.bias']
    ckpt["state_dict"]['net.transformer.layers.0.attn.qkv_proj.weight'] = ckpt["state_dict"]['transformer.layers.0.self_attn.qkv_proj.weight']
    ckpt["state_dict"]['net.transformer.layers.0.attn.qkv_proj.bias'] = ckpt["state_dict"]['transformer.layers.0.self_attn.qkv_proj.bias']
    ckpt["state_dict"]['net.transformer.layers.0.attn.o_proj.weight'] = ckpt["state_dict"]['transformer.layers.0.self_attn.o_proj.weight']
    ckpt["state_dict"]['net.transformer.layers.0.attn.o_proj.bias'] = ckpt["state_dict"]['transformer.layers.0.self_attn.o_proj.bias']
    ckpt["state_dict"]['net.transformer.layers.0.linear_net.0.weight'] = ckpt["state_dict"]['transformer.layers.0.linear_net.0.weight']
    ckpt["state_dict"]['net.transformer.layers.0.linear_net.0.bias'] = ckpt["state_dict"]['transformer.layers.0.linear_net.0.bias']
    ckpt["state_dict"]['net.transformer.layers.0.linear_net.3.weight'] = ckpt["state_dict"]['transformer.layers.0.linear_net.3.weight']
    ckpt["state_dict"]['net.transformer.layers.0.linear_net.3.bias'] = ckpt["state_dict"]['transformer.layers.0.linear_net.3.bias']
    ckpt["state_dict"]['net.transformer.layers.0.norm1.weight'] = ckpt["state_dict"]['transformer.layers.0.norm1.weight']
    ckpt["state_dict"]['net.transformer.layers.0.norm1.bias'] = ckpt["state_dict"]['transformer.layers.0.norm1.bias']
    ckpt["state_dict"]['net.transformer.layers.0.norm2.weight'] = ckpt["state_dict"]['transformer.layers.0.norm2.weight']
    ckpt["state_dict"]['net.transformer.layers.0.norm2.bias'] = ckpt["state_dict"]['transformer.layers.0.norm2.bias']
    ckpt["state_dict"]['net.output_net.0.weight'] = ckpt["state_dict"]['output_net.0.weight']
    ckpt["state_dict"]['net.output_net.0.bias'] = ckpt["state_dict"]['output_net.0.bias']
    ckpt["state_dict"]['net.output_net.2.weight'] = ckpt["state_dict"]['output_net.2.weight']
    ckpt["state_dict"]['net.output_net.2.bias'] = ckpt["state_dict"]['output_net.2.bias']

    del ckpt["state_dict"]['input_net.0.weight']
    del ckpt["state_dict"]['input_net.2.weight']
    del ckpt["state_dict"]['input_net.2.bias']
    del ckpt["state_dict"]['transformer.layers.0.self_attn.qkv_proj.weight']
    del ckpt["state_dict"]['transformer.layers.0.self_attn.qkv_proj.bias']
    del ckpt["state_dict"]['transformer.layers.0.self_attn.o_proj.weight']
    del ckpt["state_dict"]['transformer.layers.0.self_attn.o_proj.bias']
    del ckpt["state_dict"]['transformer.layers.0.linear_net.0.weight']
    del ckpt["state_dict"]['transformer.layers.0.linear_net.0.bias']
    del ckpt["state_dict"]['transformer.layers.0.linear_net.3.weight']
    del ckpt["state_dict"]['transformer.layers.0.linear_net.3.bias']
    del ckpt["state_dict"]['transformer.layers.0.norm1.weight']
    del ckpt["state_dict"]['transformer.layers.0.norm1.bias']
    del ckpt["state_dict"]['transformer.layers.0.norm2.weight']
    del ckpt["state_dict"]['transformer.layers.0.norm2.bias']
    del ckpt["state_dict"]['output_net.0.weight']
    del ckpt["state_dict"]['output_net.0.bias']
    del ckpt["state_dict"]['output_net.2.weight']
    del ckpt["state_dict"]['output_net.2.bias']

    torch.save(ckpt, model)



if __name__ == "__main__":
    if args.task == "json_to_csv":
        json_to_csv(args.json, args.csv)
    if args.task == "add_family":
        add_family(args.csv)
    if args.task == "add_total":
        add_total(args.csv)
    if args.task == "model_2_model":
        model_2_model(args.model)
logger.info(f"[+] Finished! {os.getpid()} {cmdline}")