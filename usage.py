import os
import argparse
import numpy as np
from torch import optim
import torch
from torch import nn
from utils import load_model, freeze_params
from model.config import ModelArgs


def fine_tune(max_epoch, encoder, head, optimizer, train_loader, loss_fn, gpu_id, ckpt_path):
    emb_wei = nn.Parameter(torch.tensor([0.5, 0.5], device=gpu_id), requires_grad=True)
    softmax = nn.Softmax(dim=-1)
    for epoch_idx in range(max_epoch):
        encoder.train()
        head.train()
        train_loss = 0
        train_true_label = torch.tensor([], dtype=torch.long)
        train_pred_label = torch.tensor([], dtype=torch.long)
        for batch_idx, (data, psd, label) in enumerate(train_loader):
            # shape of data: [batch size, channel number, 16, patch length]
            # shape of psd:  [batch size, channel number, 16, psd length]
            # shape of label:[batch size]
            data, psd, label = data.to(gpu_id), psd.to(gpu_id), label.to(gpu_id)
            emb, z = encoder(data, psd)
            normalized_wei = softmax(emb_wei)
            weighted_emb = emb[:, :, 0] * normalized_wei[0] + emb[:, :, 1] * normalized_wei[1]

            logit = head(weighted_emb)
            loss = loss_fn(logit, label)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_true_label = torch.cat([train_true_label, label.cpu()])
            train_pred_label = torch.cat([train_pred_label, logit.detach().cpu()], dim=0)

        print(f'Epoch {epoch_idx}, train loss = {train_loss}')

        torch.save(encoder.state_dict(), os.path.join(ckpt_path, f'encoder_{epoch_idx}.pt'))
        torch.save(head.state_dict(), os.path.join(ckpt_path, f'head_{epoch_idx}.pt'))
        torch.save(emb_wei, os.path.join(ckpt_path, f'emb_wei_{epoch_idx}.pt'))


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    torch.set_default_tensor_type(torch.FloatTensor)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_id", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_args = ModelArgs()

    class_num = None  # This is the class number of your dataset
    channel_num = None  # This is the channel number of your dataset
    encoder, head = load_model(model_args=model_args,
                               state_dict_path='path/to/the/state/dict/of/pretrained/brant2',
                               out_dim=class_num,
                               gpu_id=args.gpu_id,
                               load_pre_train=True,
                               channel_num=channel_num,
                               )
    # freeze_params() is used to freeze the parameters of Brant-2.
    # By setting `act_time` and `act_channel`, you make the parameters of the
    # last `act_time` layer(s) of the temporal encoder
    # and last `act_channel` layer(s) of the spatial encoder trainable.
    # If you want to fine-tune all the parameters, just comment out this line.
    encoder = freeze_params(model_args, encoder, act_time=2, act_channel=0)
    train_loader = None  # your train dataloader
    loss_fn = torch.nn.CrossEntropyLoss()

    # fine-tune
    if args.only_test is False:
        optimizer = optim.AdamW(
            [
                {'params': filter(lambda p: p.requires_grad, encoder.parameters()), 'lr': args.encoder_lr},
                {'params': list(head.parameters()), 'lr': args.head_lr}
            ],
            betas=(0.9, 0.95), eps=1e-5,
        )
        fine_tune(max_epoch=args.max_epoch,
                  encoder=encoder,
                  head=head,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  loss_fn=loss_fn,
                  gpu_id=args.gpu_id,
                  ckpt_path='path/to/save/your/checkpoint/'
                  )
