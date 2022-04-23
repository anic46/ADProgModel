import os
import json
import shutil
import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("brain.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    #"filename":["synthetic_split0.xls"], 
    "filename":["adni_split0.xls"],
    "datatype":["adni"],
    "algo": ["TRPO"],
    "w_lambda":[1.0],
    "gamma":[1.0],
    "cog_init":["full"],
    "cog_type":["fixed"],
    "cog_mtl":[7.0],
    "epochs":[1000],
    "batch_size":[1000],
    "eval": [True],
    "score": ["MMSE"],
    "scale":[True],
    "discount":[1.00],
    "network":[32],
    "degrade_model":["old"],
    "normalize":[False]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("configs/eval_configs/"):
    shutil.rmtree("configs/eval_configs")

os.makedirs("configs/eval_configs/")

for i, config in enumerate(all_configs):
    with open(f"configs/eval_configs/{i}.json", "w") as f:
        json.dump(config, f)
