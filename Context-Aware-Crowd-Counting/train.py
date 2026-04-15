import sys
import os

import warnings

from model import CANNet, CANNet_GRL_Frontend, CANNet_GRL_Context, CANNet_GRL_Concat

from utils import save_checkpoint

import torch
import torch.nn as nn

from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size for training')
parser.add_argument('--grl_location', type=str, default='none', choices=['none', 'frontend', 'context', 'concat'],
                    help='Location for Gradient Reversal Layer')
parser.add_argument('--lambda_domain', type=float, default=0.1,
                    help='Weight for domain classification loss')


def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 1000
    args.workers = 8
    args.seed = int(time.time())
    args.print_freq = 4
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.manual_seed(args.seed)
    print(f"Training on device: {device}")

    if args.grl_location == 'frontend':
        model = CANNet_GRL_Frontend()
    elif args.grl_location == 'context':
        model = CANNet_GRL_Context()
    elif args.grl_location == 'concat':
        model = CANNet_GRL_Concat()
    else:
        model = CANNet()

    model = model.to(device)

    criterion = nn.MSELoss(reduction='sum').to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.decay)

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True)

    history = {'train_loss': [], 'val_mae': []}

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, criterion, criterion_domain, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)

        history['train_loss'].append(train_loss)
        history['val_mae'].append(prec1)

        with open('training_history.json', 'w') as f:
            json.dump(history, f)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'best_mae': best_prec1,
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, criterion_domain, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target, country_id) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device)
        target = target.type(torch.FloatTensor).to(device)
        country_id = country_id.to(device)

        if args.grl_location != 'none':
            p = float(i + epoch * len(train_loader)) / args.epochs / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            output, domain_logits = model(img, alpha=alpha)
            output = output[:, 0, :, :]

            loss_density = criterion(output, target)
            loss_domain = criterion_domain(domain_logits, country_id)
            loss = loss_density + args.lambda_domain * loss_domain
        else:
            output = model(img)[:, 0, :, :]
            loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion):
    print('begin val')

    model.eval()

    mae = 0

    for i, (img, target, country_id) in enumerate(val_loader):
        h, w = img.shape[2:4]
        h_d = h // 2
        w_d = w // 2
        img_1 = img[:, :, :h_d, :w_d]
        img_2 = img[:, :, :h_d, w_d:]
        img_3 = img[:, :, h_d:, :w_d]
        img_4 = img[:, :, h_d:, w_d:]

        batch_img = torch.cat([img_1, img_2, img_3, img_4], dim=0).to(device)

        if args.grl_location != 'none':
            batch_density, _ = model(batch_img, alpha=1.0)
            batch_density = batch_density.data.cpu().numpy()
        else:
            batch_density = model(batch_img).data.cpu().numpy()

        pred_sum = batch_density.sum()

        mae += abs(pred_sum - target.sum())

        if i % 10 == 0 or i == len(val_loader) - 1:
            print('Validating: [{0}/{1}]\t Current MAE: {2:.3f}'.format(i, len(val_loader), mae / (i + 1)))

    mae = mae / len(val_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()