"""
Author(s):
    John Boccio
Last revision:
    3/4/2020
Description:
    A large portion of this has been taken from PyTorch's ImageNet example and changed for FER
    (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
"""

import argparse
import os
import random
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from efficientnet_pytorch import EfficientNet, utils

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import data_loader as dl
import neural_nets
from utils import FerPlusExpression
from utils import DatasetType
from utils import VisdomLinePlotter

model_names = [
    'vggface',
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
    'efficientnet-b4',
]

parser = argparse.ArgumentParser(description='PyTorch FER Training')
parser.add_argument('-a', '--arch', dest='arch', metavar='ARCH', default='vggface',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vggface)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='model.pth', type=str, metavar='save-path', dest='save_path',
                    help='path to save the training data (default: model.pth)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_false',
                    help='run training with early stopping enabled')
parser.add_argument('--patience', default=20, type=int, dest='patience', metavar='patience',
                    help='patience to use for early stopping if it is enabled')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='plot training progress using visdom')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.visdom:
        plotter = VisdomLinePlotter(env_name="FER Training")
    else:
        plotter = None

    # create model and get transformations
    if args.arch == 'vggface':
        model = neural_nets.VggVdFaceFerDag()
        train_transform = None
        val_transform = transforms.Compose(
            [  # image_processing.crop_face_transform,
                transforms.Resize(model.meta["imageSize"][0]),
                transforms.ToTensor(),
                lambda x: x * 255,
                transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])])
        if args.evaluate is not True:
            warnings.warn("Cannot train vggface, changing settings to evaluation mode")
            args.evaluate = True
    elif args.arch.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(args.arch, in_channels=1, num_classes=len(FerPlusExpression))
        res = utils.efficientnet_params(args.arch)[2]
        train_transform = transforms.Compose(
            [transforms.Resize(res),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.449], [0.226])]
        )
        val_transform = transforms.Compose(
            [transforms.Resize(res),
             transforms.ToTensor(),
             transforms.Normalize([0.449], [0.226])]
        )
    else:
        warnings.warn("Invalid architecture selected: ", args.arch)
        train_transform = None
        val_transform = None
        return

    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=.1)

    best_acc = 0
    start_epoch = 0
    train_losses = []
    val_losses = []
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_sched.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_set = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=train_transform)
    val_set = dl.FER2013Dataset(set_type=DatasetType.VALIDATION, tf=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, conf_mat=True)
        return

    early_stop = EarlyStopping(patience=args.patience)
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        train_losses.append(loss)
        lr_sched.step(epoch)

        # evaluate on validation set
        loss, val_acc = validate(val_loader, model, criterion, args)
        val_losses.append(loss)

        if plotter is not None:
            plotter.plot('loss', 'train', 'Loss', epoch, train_losses[-1])
            plotter.plot('loss', 'val', 'Loss', epoch, val_losses[-1])
            plotter.plot('acc', 'train', 'Accuracy', epoch, train_acc)
            plotter.plot('acc', 'val', 'Accuracy', epoch, val_acc)

        # remember best acc and save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_sched.state_dict(),
            }
            torch.save(save_checkpoint, args.save_path)

        save_checkpoint = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_sched.state_dict(),
        }
        torch.save(save_checkpoint, "last_" + args.save_path)

        if args.early_stopping:
            early_stop(val_acc)
            if early_stop.early_stop:
                print("Early Stopping detected: Ending training and saving model")
                break


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        images = sample['img']
        target = sample['expression']
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        a = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        acc.update(a, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, acc.avg


def validate(val_loader, model, criterion, args, conf_mat=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix='Test: ')
    if conf_mat:
        mat = ConfusionMat()
    else:
        mat = None

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            images = sample['img']
            target = sample['expression']

            if torch.cuda.is_available():
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            a = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            acc.update(a, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if mat is not None:
                _, predicted = torch.max(output.data, 1)
                a = [FerPlusExpression(i).name for i in target.tolist()]
                p = [FerPlusExpression(i).name for i in predicted.tolist()]
                mat.update(a, p, extend=True)

            if i % args.print_freq == 0:
                progress.display(i)

    if mat is not None:
        mat.save()

    return losses.avg, acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ConfusionMat(object):
    def __init__(self):
        self.actual = []
        self.pred = []

    def save(self):
        mat = sklearn.metrics.confusion_matrix(self.actual, self.pred)
        display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=mat,
                                                         display_labels=[n for n, _ in vars(FerPlusExpression).items()])
        display = display.plot()
        plt.savefig("conf_mat.png")

    def update(self, actual, pred, extend=False):
        if extend:
            self.actual.extend(actual)
            self.pred.extend(pred)
        else:
            self.actual.append(actual)
            self.pred.append(pred)


class EarlyStopping:
    """ Modified version of https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py """
    """ Early stops the training if validation loss doesn't improve after a given patience. """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    return correct / output.size(0)


if __name__ == '__main__':
    main()
