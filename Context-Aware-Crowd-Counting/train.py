import sys
import os

import warnings

from model import CANNet, CANNet_GRL_Frontend, CANNet_GRL_Context, CANNet_GRL_Concat, CANNet_CORAL_Frontend, CANNet_CORAL_Context, CANNet_CORAL_Concat

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
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size for training')
parser.add_argument('--algorithm', type=str, default='grl', choices=['grl', 'coral', 'none'],
                    help='Domain generalization algorithm to use')
parser.add_argument('--grl_location', type=str, default='none', choices=['none', 'frontend', 'context', 'concat'],
                    help='Location for Gradient Reversal Layer')
parser.add_argument('--coral_location', type=str, default='none', choices=['none', 'frontend', 'context', 'concat'],
                    help='Location for CORAL feature extraction')
parser.add_argument('--lambda_domain', type=float, default=0.0001,
                    help='Weight for domain classification loss')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--workers', type=int, default=12,
                    help='number of data loading workers')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='limit the number of batches to process for quick testing')
parser.add_argument('--skip_val', action='store_true',
                    help='skip validation loop')
parser.add_argument('--exp_name', type=str, default='',
                    help='Experiment name suffix for saving files')


# Main orchestrator: parses command line flags, initializes model, sets up datasets/loaders, and runs training/validation loops
def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.seed = int(time.time())
    args.print_freq = 4
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.manual_seed(args.seed)
    print(f"Training on device: {device}")

    if args.algorithm == 'coral':
        if args.coral_location == 'frontend':
            model = CANNet_CORAL_Frontend()
        elif args.coral_location == 'context':
            model = CANNet_CORAL_Context()
        elif args.coral_location == 'concat':
            model = CANNet_CORAL_Concat()
        else:
            model = CANNet()
    else:
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

    history = {'val_mae': [], 'train_loss': [], 'val_loss': [], 'val_density_loss': [], 'val_domain_loss': []}

    for epoch in range(args.start_epoch, args.epochs):
        _, train_loss, train_domain_loss = train(train_loader, model, criterion, criterion_domain, optimizer, epoch)
        
        if not args.skip_val:
            val_mae, val_rmse, val_density_loss, val_domain_loss = validate(val_loader, model, criterion, criterion_domain)
        else:
            val_mae, val_rmse, val_density_loss, val_domain_loss = 0, 0, 0, 0

        history['val_mae'].append(val_mae)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_density_loss + val_domain_loss)
        history['val_density_loss'].append(val_density_loss)
        history['val_domain_loss'].append(val_domain_loss)

        history_file = f'training_history_{args.exp_name}.json' if args.exp_name else 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(history, f)

        is_best = val_mae < best_prec1
        best_prec1 = min(val_mae, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        checkpoint_file = f'checkpoint_{args.exp_name}.pth.tar' if args.exp_name else 'checkpoint.pth.tar'
        save_checkpoint({
            'best_mae': best_prec1,
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, is_best, filename=checkpoint_file)

        # Plot learning curves midway through and at the end of training
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss', marker='o', markersize=3)
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Total Loss', color='green', marker='o', markersize=3)
        plt.title('Train & Val Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['val_density_loss']) + 1), history['val_density_loss'], label='Val Density Loss', color='purple', marker='o', markersize=3)
        plt.plot(range(1, len(history['val_domain_loss']) + 1), history['val_domain_loss'], label='Val Domain Loss', color='orange', marker='o', markersize=3)
        plt.title('Validation Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        curves_file = f'learning_curves_{args.exp_name}.png' if args.exp_name else 'learning_curves_updated.png'
        plt.savefig(curves_file, dpi=300)
        plt.close()
        print(f"Learning curves saved to '{curves_file}'")

# Computes CORAL loss by finding the Frobenius norm distance between covariance matrices of source and target features
def calc_coral_loss(source, target):
    d = source.size(1)
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (source.size(0) - 1 + 1e-8)
    
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (target.size(0) - 1 + 1e-8)
    
    # frobenius norm
    loss = torch.mean((xc - xct)**2)
    return loss



def train(train_loader, model, criterion, criterion_domain, optimizer, epoch):
    losses_density = AverageMeter()
    losses_domain = AverageMeter()
    maes = AverageMeter()
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

        if args.algorithm == 'coral' and args.coral_location != 'none':
            output, features = model(img)
            output = output[:, 0, :, :]

            loss_density = criterion(output, target)
            
            unique_domains = torch.unique(country_id)
            coral_l = 0.0
            pairs = 0
            
            for d_i in range(len(unique_domains)):
                for d_j in range(d_i + 1, len(unique_domains)):
                    f_i = features[country_id == unique_domains[d_i]]
                    f_j = features[country_id == unique_domains[d_j]]
                    if f_i.size(0) > 1 and f_j.size(0) > 1:
                        coral_l += calc_coral_loss(f_i, f_j)
                        pairs += 1
            
            if pairs > 0:
                loss_domain = (coral_l / pairs) * args.lambda_domain
            else:
                loss_domain = torch.tensor(0.0, device=device, requires_grad=True)
                
            loss = loss_density + loss_domain
            losses_density.update(loss_density.item(), img.size(0))
            losses_domain.update(loss_domain.item() if isinstance(loss_domain, torch.Tensor) else loss_domain, img.size(0))

        elif args.grl_location != 'none':
            p = float(i + epoch * len(train_loader)) / args.epochs / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            combined_alpha = alpha

            output, domain_logits = model(img, alpha=combined_alpha)
            output = output[:, 0, :, :]

            loss_density = criterion(output, target)
            loss_domain = criterion_domain(domain_logits, country_id)
            
            weighted_domain_loss = args.lambda_domain * loss_domain
            loss = loss_density + weighted_domain_loss
            losses_density.update(loss_density.item(), img.size(0))
            losses_domain.update(weighted_domain_loss.item(), img.size(0))
        else:
            output = model(img)[:, 0, :, :]
            loss_density = criterion(output, target)
            loss_domain = torch.tensor(0.0).to(device)
            loss = loss_density
            losses_density.update(loss_density.item(), img.size(0))
            losses_domain.update(0.0, img.size(0))

        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping to prevent adversarial collapse explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss Density {loss_density.val:.4f} ({loss_density.avg:.4f})\t'
                  'Loss Domain {loss_domain.val:.4f} ({loss_domain.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss_density=losses_density, loss_domain=losses_domain))

        if args.max_batches > 0 and i >= args.max_batches:
            break

    return maes.avg, losses_density.avg, losses_domain.avg



def validate(val_loader, model, criterion, criterion_domain):
    print('begin val')

    model.eval()

    mae = 0
    rmse_sum = 0
    losses_density = AverageMeter()
    losses_domain = AverageMeter()

    with torch.no_grad():
        for i, (img, target, country_id) in enumerate(val_loader):
            h, w = img.shape[2:4]
            h_d = h // 2
            w_d = w // 2
            img_1 = img[:, :, :h_d, :w_d]
            img_2 = img[:, :, :h_d, w_d:]
            img_3 = img[:, :, h_d:, :w_d]
            img_4 = img[:, :, h_d:, w_d:]
    
            batch_img = torch.cat([img_1, img_2, img_3, img_4], dim=0).to(device)
    
            if args.algorithm == 'coral' and args.coral_location != 'none':
                batch_density, _ = model(batch_img, alpha=1.0)
                domain_logits = None
            elif args.grl_location != 'none':
                batch_density, domain_logits = model(batch_img, alpha=1.0)
            else:
                batch_density = model(batch_img)
                domain_logits = None
                
            h_out_d = batch_density.shape[2]
            w_out_d = batch_density.shape[3]
            
            pred_density = torch.zeros(1, h_out_d * 2, w_out_d * 2, device=device)
            pred_density[0, :h_out_d, :w_out_d] = batch_density[0, 0]
            pred_density[0, :h_out_d, w_out_d:] = batch_density[1, 0]
            pred_density[0, h_out_d:, :w_out_d] = batch_density[2, 0]
            pred_density[0, h_out_d:, w_out_d:] = batch_density[3, 0]
            
            target = target.type(torch.FloatTensor).to(device)
        
            if pred_density.shape[1:] != target.shape[1:]:
                import torch.nn.functional as F
                pred_density = F.interpolate(pred_density.unsqueeze(0), size=(target.shape[1], target.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
                ratio = (target.shape[1] * target.shape[2]) / (pred_density.shape[1] * pred_density.shape[2])
                pred_density = pred_density * ratio
            
            loss_density = criterion(pred_density, target)
            losses_density.update(loss_density.item(), 1)
            
            if domain_logits is not None:
                expanded_country_id = country_id.to(device).repeat(4)
                loss_domain = criterion_domain(domain_logits, expanded_country_id)
                losses_domain.update((args.lambda_domain * loss_domain).item(), 4)
            else:
                losses_domain.update(0.0, 4)
            
            batch_density_np = batch_density.data.cpu().numpy()
            pred_sum = batch_density_np.sum()
            target_sum = target.sum().item()
            
            err = pred_sum - target_sum
    
            mae += float(abs(err))
            rmse_sum += float(err ** 2)
    
            if i % 10 == 0 or i == len(val_loader) - 1:
                print('Validating: [{0}/{1}]\t Current MAE: {2:.3f}\t Density Loss: {3:.4f}\t Domain Loss: {4:.4f}'.format(
                    i, len(val_loader), mae / (i + 1), losses_density.avg, losses_domain.avg))

    mae = mae / len(val_loader)
    rmse = np.sqrt(rmse_sum / len(val_loader))
    
    print(' * MAE {mae:.3f} * RMSE {rmse:.3f} * Density Loss {loss_dens:.4f} * Domain Loss {loss_dom:.4f} '
          .format(mae=mae, rmse=rmse, loss_dens=losses_density.avg, loss_dom=losses_domain.avg))

    return mae, rmse, losses_density.avg, losses_domain.avg


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