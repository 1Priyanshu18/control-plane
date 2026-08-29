import random
from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    random.seed(cfg["seed"])
    return cfg
