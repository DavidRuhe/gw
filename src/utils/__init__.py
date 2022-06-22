import torch
import os
import random
import numpy as np


def set_seed(seed: int, workers: bool = False):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    if workers:
        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
