import os
import pickle
import random
import time

import numpy as np
import torch
from torch._C._distributed_c10d import ReduceOp
from torch.utils.data import DataLoader

from config import TrainArgs
import torch.nn as nn
from utils import PreDataset, data_preprocess, master_print, master_save, format_time, RandomVariableForTraining
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch import autocast
from torch.cuda.amp import GradScaler


def do_epoch(args: TrainArgs, dataloader, brant2, head, optimizer, scheduler, scaler, loss_fn):
    tot_loss_m = 0
    tot_loss_f = 0
    iters = 0
    for batch_idx, (batch_data) in enumerate(dataloader):
        lb_data = batch_data[0].to(args.local_rank)
        lb_psd = batch_data[1].to(args.local_rank)
        fore_data = batch_data[2].to(args.local_rank)

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
            lb_data_copy = lb_data.detach().clone()
            # obtain data shape
            bsz, ch_num, seq_len, patch_len = lb_data.shape

            _, z = brant2(lb_data, lb_psd)
            # mask_pred: [bsz, ch_num, seq_len, patch_len], forecast_pred: [bsz, ch_num, seq_len, forecast_len]
            mask_pred, forecast_pred = head[str(patch_len)](z)
            loss_m = loss_fn(mask_pred, lb_data_copy) / args.accumulation_step
            loss_f = loss_fn(forecast_pred, fore_data) / args.accumulation_step

            loss = loss_m + loss_f

        scaler.scale(loss).backward()
        if (batch_idx+1) % args.accumulation_step == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        tot_loss_m += loss_m.float().item()
        tot_loss_f += loss_f.float().item()
        iters += 1

    return tot_loss_m, tot_loss_f, iters // args.accumulation_step


def pre_train(args: TrainArgs, brant2, head, optimizer, scheduler, rand_var_for_tr: RandomVariableForTraining):

    loss_fn = nn.MSELoss(reduction='mean')
    scaler = GradScaler(enabled=args.amp)

    for epoch_idx in range(args.last_epoch+1, args.max_epoch):
        brant2.train()
        head.train()
        epoch_loss_m = 0
        epoch_loss_f = 0
        epoch_iters = 0
        process_data_time, train_time = 0, 0  # calculate the time for processing data and training data
        _rng = rand_var_for_tr.rng[epoch_idx]
        train_rv = [rand_var_for_tr.train_rv[i] for i in _rng]
        for rv in train_rv:
            # rv: (file path, augmentation rate, channel number)
            read_data_start = time.time()
            data = np.load(rv[0])
            lb_data, fore_data = data_preprocess(args, data, rv[1], rv[2])
            dataset = PreDataset(lb_data, fore_data, block=args.block, rank=args.local_rank)
            bsz = args.batch_size // lb_data.shape[0]
            if dist.is_initialized():
                sampler = DistributedSampler(dataset,
                                             num_replicas=args.world_size,
                                             rank=args.local_rank,
                                             shuffle=True, drop_last=True)
                dataloader = DataLoader(dataset,
                                        batch_size=bsz,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        pin_memory=True,
                                        sampler=sampler)
                dataloader.sampler.set_epoch(epoch_idx)
            else:
                dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=False)

            # the end of processing data
            process_data_time += time.time() - read_data_start

            train_start = time.time()
            loss_m, loss_f, iters = do_epoch(args, dataloader, brant2, head, optimizer, scheduler, scaler, loss_fn)
            epoch_loss_m += loss_m
            epoch_loss_f += loss_f
            epoch_iters += iters

            # the end of training
            train_time += time.time() - train_start

        epoch_loss_m = torch.tensor(epoch_loss_m, dtype=torch.float32).to(args.local_rank)
        dist.all_reduce(epoch_loss_m, op=ReduceOp.SUM)
        epoch_loss_f = torch.tensor(epoch_loss_f, dtype=torch.float32).to(args.local_rank)
        dist.all_reduce(epoch_loss_f, op=ReduceOp.SUM)
        epoch_iters = torch.tensor(epoch_iters, dtype=torch.long).to(args.local_rank)
        dist.all_reduce(epoch_iters, op=ReduceOp.SUM)

        master_print(f'Epoch {epoch_idx} spent {format_time(int(process_data_time))} for processing data and {format_time(int(train_time))} for training on GPU\n'
                     f'mask loss = {epoch_loss_m / epoch_iters}\n'
                     f'forecast loss = {epoch_loss_f / epoch_iters}')

        master_save(brant2.state_dict(), os.path.join(args.ckpt_path, f'brant2_{epoch_idx}.pt'))
        master_save(head.state_dict(), os.path.join(args.ckpt_path, f'head_{epoch_idx}.pt'))
        master_save(optimizer.state_dict(), os.path.join(args.ckpt_path, f'optim_{epoch_idx}.pt'))
        master_save(scheduler.state_dict(), os.path.join(args.ckpt_path, f'sche_{epoch_idx}.pt'))
