### Dataset格式转换，from json to csv
import os
import argparse
from loguru import logger
import pandas as pd
import psutil
from tqdm import tqdm
import json
import collections
import datatable as dt

cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[$] {os.getpid()} {cmdline}")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str)
parser.add_argument("--json", type=str)
parser.add_argument("--csv", type=str)
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
    
if __name__ == "__main__":
    if args.task == "json_to_csv":
        json_to_csv(args.json, args.csv)
    if args.task == "add_family":
        add_family(args.csv)

logger.info(f"[+] Finished! {os.getpid()} {cmdline}")