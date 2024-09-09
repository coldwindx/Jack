### Dataset格式转换，from json to csv
import os
import argparse
from loguru import logger
import psutil
from tqdm import tqdm
import json
import collections
import datatable as dt

cmdline = " ".join(psutil.Process(os.getpid()).cmdline())
logger.info(f"[$] {os.getpid()} {cmdline}")

parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str)
parser.add_argument("--csv", type=str)
args = parser.parse_args()

### read json data
collect = collections.defaultdict(list)
for file in args.json.split(","):
    logger.info(f"[+] load {file} ...")
    with open(file, "r") as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            for k, v in data.items():
                collect[k].append(v)

### transform json to csv
df = dt.Frame(collect)
df.to_csv(args.csv)

logger.info(f"[+] Finished! {os.getpid()} {cmdline}")