import datetime
import os
import argparse
import sys

import pickle
from torch import nn

import time
import numpy as np
import torch
import random
import wandb
import torch.distributed as dist
from config import TrainArgs

from utils import get_model, get_scheduler, get_optimizer, master_print, print_args, RandomVariableForTraining
from model.config import ModelArgs
from pretrain import pre_train


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    master_print('This progress began at: ' + time.asctime(time.localtime(time.time())))
    torch.set_default_tensor_type(torch.FloatTensor)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)

    parser.add_argument('--dist_data_parallel', type=str2bool, default=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dev_ids', type=str, default="0,1,2,3")

    parser.add_argument("--amp", type=str2bool, default=False,
                        help='whether to use automatic mixed precision')
    parser.add_argument("--last_epoch", type=int, default=-1,
                        help='load the model from last train, -1 means a new model')
    parser.add_argument("--max_file_per_pat", type=int, default=1)
    parser.add_argument("--use_tueg", type=str2bool, default=True)
    parser.add_argument("--tueg_file_ratio", type=float, default=0.1)
    parser.add_argument("--add_files", type=int, default=0)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--time_n_layers", type=int, default=10)
    parser.add_argument("--channel_n_layers", type=int, default=2)

    parser.add_argument('--aug_rate_list', nargs='+', type=float, default=[2, 1, 0.5, 0.25, 0.125])

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.dev_ids
    os.environ['NCCL_P2P_DISABLE'] = '1'  # solve the process stuck before training with gpu hang at 100% utilization (in ddp mode with 'nccl' backend)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    train_args = TrainArgs()
    model_args = ModelArgs()

    if args.dist_data_parallel:
        dist.init_process_group(backend="nccl")
        train_args.local_rank = args.local_rank
        train_args.world_size = len(args.dev_ids.split(','))
        torch.cuda.set_device(args.local_rank)
    else:
        train_args.local_rank = args.gpu_id
        train_args.world_size = 1

    train_args.dist_data_parallel = args.dist_data_parallel
    train_args.aug_rate_list = args.aug_rate_list
    train_args.last_epoch = args.last_epoch
    train_args.amp = args.amp

    model_args.seq_len = train_args.look_back_token
    model_args.patch_len = train_args.patch_len
    model_args.d_model = args.d_model
    model_args.time_n_layers = args.time_n_layers
    model_args.channel_n_layers = args.channel_n_layers
    model_args.ff_hidden = args.d_model * 3

    train_args.ckpt_path = os.path.join(train_args.ckpt_path, f'{model_args.d_model}-{model_args.time_n_layers}-{model_args.channel_n_layers}')

    # This line of code CANNOT run in DDP mode.
    # You can run this line before then the random variable will be saved, then restart the program in DDP mode.
    rand_var_for_tr = RandomVariableForTraining(train_args)

    master_print('Train args:')
    print_args(train_args)

    master_print('Model args:')
    print_args(model_args)

    brant2, head, optimizer, scheduler = get_model(model_args, train_args)

    pre_train(args=train_args,
              brant2=brant2,
              head=head,
              optimizer=optimizer,
              scheduler=scheduler,
              rand_var_for_tr=rand_var_for_tr)

    master_print('\n' * 3)
    master_print('─' * 50)
    master_print('This progress finished at: ' + time.asctime(time.localtime(time.time())))
