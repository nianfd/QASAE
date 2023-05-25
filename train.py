import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import argparse
import os
import shutil
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from light_dnn import gau_trans
from load_filelist import FileListDataLoader

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model





parser = argparse.ArgumentParser(description='PyTorch Light DNN Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=13, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='./data/npydata/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='./data/train_v1.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='./data/test_v1.txt', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='./model/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
import sys
#sys.stdout = open("train_log_diff512.txt", "w")
def main():
    global args
    args = parser.parse_args()

    model = gau_trans()


    if args.cuda:
        model = model.cuda()

    #f = open('train_log.txt', 'w')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #load image
    train_loader = torch.utils.data.DataLoader(
        FileListDataLoader(root=args.root_path, fileList=args.train_list),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        FileListDataLoader(root=args.root_path, fileList=args.val_list),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function and optimizer
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss()

    if args.cuda:
        criterion.cuda()

    validate(val_loader, model, criterion)
    bestValidloss = 100000
    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        validloss = validate(val_loader, model, criterion)

        #save_name = args.save_path + 'lttb_conv_transformer_' + str(epoch+1) + '_checkpoint.pth.tar'
        if(validloss<bestValidloss):
            bestValidloss = validloss
            save_name = args.save_path + '256_conv_' + 'best' + '_checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, save_name)
            print('best val loss:')
            print(bestValidloss)

    print('Total best val loss:')
    print(bestValidloss)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, input_diff, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.cuda:
            input   = input.cuda()
            input_diff = input_diff.cuda()
            target  = target.cuda()

        # compute output
        #print(input)
        #print("*********************************")
        output = model(input, input_diff)
        #print(output)
        #print(target)
        loss   = criterion(output, target)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    losses     = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    print('-------------------------')
    for i, (input, input_diff, target) in enumerate(val_loader):
        if args.cuda:
            input = input.cuda()
            input_diff = input_diff.cuda()
            target = target.cuda()

        print('gt:')
        print(target)
        # compute output
        output = model(input, input_diff)
        print('pd:')
        print(output)
        loss   = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    print('\nTest set: Average loss: {}\n'.format(losses.avg))

    return losses.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    scale = 0.5
    step  = 40
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

if __name__ == '__main__':
    main()


