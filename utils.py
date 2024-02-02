import os
import random

import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import Dataset
import torch.optim as optim
from lion_pytorch import Lion
from model.config import ModelArgs
from model.encoder import Brant2Encoder
from model.utils import PredictHead
from config import TrainArgs
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import asdict
from transformers import get_cosine_schedule_with_warmup
from scipy.interpolate import interp1d


class RandomVariableForTraining:
    def __init__(self, args: TrainArgs):
        train_rv_path = os.path.join(args.ckpt_path, f'train_rv.pkl')
        rng_path = os.path.join(args.ckpt_path, f'rng.npy')

        # if random variables are already generated
        if os.path.exists(train_rv_path) and os.path.exists(rng_path):
            with open(train_rv_path, 'rb') as f:
                self.train_rv = pickle.load(f)
            self.rng = np.load(rng_path)
        # generate random variables
        else:
            # get data files
            if not os.path.exists(args.ckpt_path):
                os.makedirs(args.ckpt_path)
            # The paths of training files saved in .pkl
            file_path = os.path.join(args.ckpt_path, f'files.pkl')
            file = open(file_path, 'rb')
            data_files = pickle.load(file)

            # generate rv for data files : (file path, augmentation rate, channel number),
            random.shuffle(data_files)
            single_channel_file_num = int(args.single_channel_ratio*len(data_files))
            single_channel_files = data_files[:single_channel_file_num]
            multiple_channel_files = data_files[single_channel_file_num:]

            rv = []
            for file in single_channel_files:
                aug_rate = random.choice(args.aug_rate_list)
                rv.append((file, aug_rate, 1))
            channel_list = [2, 4, 8, 16, 32, 64, 128]
            for file in multiple_channel_files:
                aug_rate = random.choice(args.aug_rate_list)
                split_channel_num = random.choice(channel_list)
                rv.append((file, aug_rate, split_channel_num))

            train_file_num = len(rv)
            self.train_rv = rv
            # rng is used to generate the order of files in advance to ensure synchronization between each GPU in DDP mode.
            self.rng = np.zeros((args.max_epoch, train_file_num), dtype=int)
            for i in range(args.max_epoch):
                self.rng[i] = np.random.permutation(train_file_num)

            with open(train_rv_path, 'wb') as f:
                pickle.dump(self.train_rv, f)

            np.save(rng_path, self.rng)


def data_preprocess(args: TrainArgs, data, aug_rate, split_channel_num):
    ch_num, ch_len = data.shape

    seq_pts = args.patch_len * (args.look_back_token + args.fore_token)
    seq_num = ch_len // seq_pts
    if seq_num <= 1:
        return None, None
    # truncate
    data = data[:, :seq_num * seq_pts]
    data = data.reshape(ch_num, seq_num, seq_pts)

    gap = int(1 / aug_rate)
    new_patch_len = int(aug_rate * args.patch_len)
    # resampling
    if gap > 1:
        data = data[:, :, ::gap]
    elif gap < 1:
        func = interp1d(np.arange(seq_pts), data, kind='cubic', axis=2)
        data = func(np.linspace(0, seq_pts-1, int(aug_rate * seq_pts)))

    if ch_num > split_channel_num > 1:
        split_data = split_channels(data, split_channel_num)
        data = np.concatenate(split_data, axis=1)

    lb_data = data[:, :, :new_patch_len * args.look_back_token].reshape(split_channel_num, -1, args.look_back_token, new_patch_len)
    fore_data = data[:, :, new_patch_len * args.look_back_token:].reshape(split_channel_num, -1, new_patch_len * args.fore_token)

    return lb_data, fore_data


def split_channels(data, channel_num):
    tot_channel_num, _, _ = data.shape

    perm = np.random.permutation(tot_channel_num)
    split_num = tot_channel_num // channel_num

    split_data = []
    for split in range(split_num):
        idx = perm[split*channel_num:(split+1)*channel_num]
        split_data.append(data[idx])

    return split_data


def periodogram(X: torch.Tensor, fs=256, detrend=False, scaling='density'):
    if X.dim() > 2:
        X = torch.squeeze(X)
    elif X.dim() == 1:
        X = X.unsqueeze(0)

    if detrend:
        X -= X.mean(-1, keepdim=True)

    N = X.size(-1)
    assert N % 2 == 0

    df = fs / N
    f = torch.arange(0, N / 2 + 1) * df

    dual_side = torch.fft.fft(X)
    half_idx = int(N / 2 + 1)
    single_side = dual_side[:, 0:half_idx]
    win = torch.abs(single_side)

    ps = win ** 2
    if scaling == 'density':
        scale = N * fs
    elif scaling == 'spectrum':
        scale = N ** 2
    elif scaling is None:
        scale = 1
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    Pxy = ps / scale

    Pxy[:, 1:-1] *= 2
    Pxy = Pxy.cpu()

    return f, Pxy.squeeze()


class PreDataset(Dataset):
    def  __init__(self, lb_data, fore_data, block, rank):
        super(PreDataset, self).__init__()
        self.block = block
        self.rank = rank
        self.lb_data = lb_data
        self.fore_data = fore_data
        # lb_data: [channel_num, seq_num, seq_len, patch_len]
        # fore_data: [channel_num, seq_num, fore_len]

        ch_num, self.seq_num, seq_len, patch_len = lb_data.shape

        lb_token_num = ch_num * self.seq_num * seq_len
        lb_block_num = lb_token_num // block + 1
        _lb_data = lb_data.reshape(lb_token_num, patch_len)
        self.lb_psd = self.cal_psd(_lb_data, lb_block_num, patch_len // 2 + 1)
        self.lb_psd = self.lb_psd.reshape(ch_num, self.seq_num, seq_len, -1)

    def cal_psd(self, data, block_num, psd_length):
        token_num, _ = data.shape
        psd = torch.zeros(token_num, psd_length)
        for b in range(block_num):
            _, _psd = periodogram(torch.from_numpy(data[b*self.block:(b+1)*self.block]).to(self.rank), fs=256)
            psd[b*self.block:(b+1)*self.block] = _psd
        return psd

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):
        return self.lb_data[:, idx], self.lb_psd[:, idx], self.fore_data[:, idx]


def master_save(state_dict, path):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            torch.save(state_dict, path)
    else:
        torch.save(state_dict, path)


def master_print(str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(str, flush=True)
    else:
        print(str, flush=True)


def print_args(args):
    args_dict = asdict(args)
    for k ,v in args_dict.items():
        if 'path' not in k and 'list' not in k and 'files' not in k:
            master_print(str(k) + ': ' + str(v))
    master_print('\n')


def get_model(model_args: ModelArgs, train_args: TrainArgs):
    head = nn.ModuleDict()
    patch_len_list = [int(train_args.patch_len * aug_rate) for aug_rate in train_args.aug_rate_list]
    for patch_len in patch_len_list:
        forecast_len = patch_len * train_args.fore_token
        head[str(patch_len)] = PredictHead(model_args, patch_len, forecast_len).to(train_args.local_rank)

    # load last model ckpt
    if train_args.load_his_ckpt and train_args.last_epoch >= 0:
        brant2_state_dict = torch.load(os.path.join(train_args.ckpt_path, f'brant2_{train_args.last_epoch}.pt'), map_location=f'cuda:{train_args.local_rank}')
        head_state_dict = torch.load(os.path.join(train_args.ckpt_path, f'head_{train_args.last_epoch}.pt'), map_location=f'cuda:{train_args.local_rank}')
        brant2 = Brant2Encoder(model_args, do_mask=True).to(train_args.local_rank)
        brant2.load_state_dict(brant2_state_dict)
        head.load_state_dict(head_state_dict)

        optimizer = get_optimizer(optimizer_name=train_args.optimizer, brant2=brant2, head=head, lr=train_args.lr)
        scheduler = get_scheduler(optimizer=optimizer, scheduler_name=train_args.scheduler, lr=train_args.lr)
        optim_state_dict = torch.load(os.path.join(train_args.ckpt_path, f'optim_{train_args.last_epoch}.pt'), map_location=f'cuda:{train_args.local_rank}')
        sche_state_dict = torch.load(os.path.join(train_args.ckpt_path, f'sche_{train_args.last_epoch}.pt'), map_location=f'cuda:{train_args.local_rank}')
        optimizer.load_state_dict(optim_state_dict)
        scheduler.load_state_dict(sche_state_dict)
    else:
        brant2 = Brant2Encoder(model_args, do_mask=True).to(train_args.local_rank)
        optimizer = get_optimizer(optimizer_name=train_args.optimizer, brant2=brant2, head=head, lr=train_args.lr)
        scheduler = get_scheduler(optimizer=optimizer, scheduler_name=train_args.scheduler, lr=train_args.lr)

    if train_args.dist_data_parallel:
        brant2 = DDP(brant2, device_ids=[train_args.local_rank], output_device=[train_args.local_rank]).module
        head = DDP(head, device_ids=[train_args.local_rank], output_device=[train_args.local_rank]).module

    brant2_params = sum(p.numel() for p in brant2.parameters())
    master_print(f'Model Params: {brant2_params / 1e6}M')

    return brant2, head, optimizer, scheduler


def get_optimizer(optimizer_name, brant2, head, lr):
    param_dict_list = [
        {'params': list(brant2.parameters()), 'lr': lr},
        {'params': list(head.parameters()), 'lr': lr},
    ]
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_dict_list,
            betas=(0.9, 0.99), eps=1e-8,
        )
    elif optimizer_name == 'adamw':  # the default optimizer
        optimizer = optim.AdamW(
            param_dict_list,
            betas=(0.9, 0.95), eps=1e-5,
        )
    elif optimizer_name == 'lion':
        optimizer = Lion(
            param_dict_list,
            betas=(0.9, 0.99), weight_decay=1e-2,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer, scheduler_name, lr):
    if scheduler_name == 'mul_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif scheduler_name == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1*lr, max_lr=lr, step_size_up=3000, cycle_momentum=False)
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.1*lr)
    elif scheduler_name == 'cosinewithwarmup':  # the default scheduler
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=150000,)
    else:
        raise NotImplementedError

    return scheduler


def load_data(data_file):
    if ',' in data_file:
        data_file, label_file = data_file.split(',')
        data = delete_neg_label(data_file, label_file)
    else:
        data = np.load(data_file)

    return data


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours}h {minutes}min {seconds}s"
