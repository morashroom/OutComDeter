# encoding=utf-8
import fire

import two_stage
from eval import eval_from_config
from infer import infer_from_config
from train import train_from_config


OCD_CFG = "configs/OC-Hunter.yml"
OCD_DIR = "OC-Hunter"



def run_oc_hunter(cfg=OCD_CFG,
            log_dir=OCD_DIR):
    train_from_config(cfg, log_dir)
    infer_from_config(cfg, log_dir)
    result = eval_from_config(cfg, log_dir)
    print(result)


if __name__ == '__main__':
    fire.Fire({
        "run_oc_hunter": run_oc_hunter,
    })
