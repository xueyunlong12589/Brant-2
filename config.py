import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pickle


@dataclass
class TrainArgs:
    max_epoch: int = 100
    batch_size: int = 512

    last_epoch: int = -1
    load_his_ckpt: bool = True
    lr: float = 1e-5
    optimizer: str = 'adamw'
    scheduler: str = 'cosinewithwarmup'

    world_size: int = 0
    local_rank: int = 0

    accumulation_step: int = 16
    num_workers: int = 4
    amp: bool = False
    dist_data_parallel: bool = True

    aug_rate_list: Tuple[float] = None

    patch_len: int = 1024
    look_back_token: int = 16
    fore_token: int = 4
    single_channel_ratio: float = 0.3

    block: int = 2000000
    ckpt_path: str = '/path/of/model_ckpt'








