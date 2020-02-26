"""
Author(s):
    John Boccio
Last revision:
    2/25/2020
Description:
    Place for common classes and functions among the FER datasets.
"""
import argparse
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import torch


class Expression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


def get_fer_parser():
    parser = argparse.ArgumentParser(description="PyTorch training for Facial Expression Recognition using transfer "
                                                 "learning and FER2013")
    parser.add_argument('--eval', action='store_true', dest='eval',
                        help="Only perform validation using the validation set and do not train")
    parser.add_argument('--lr', metavar="lr", type=float, default=0.05, dest='lr',
                        help="Specify the learning rate for training (default 0.05)")
    parser.add_argument('--momentum', metavar="momentum", type=float, default=0.9, dest='momentum',
                        help="Momentum for SGD (default 0.9)")
    parser.add_argument('--weight-decay', metavar="weight-decay", type=float, default=0.001, dest='weight_decay',
                        help="Weight decay for SGD (default 0.001)")
    parser.add_argument('--max-epochs', metavar="epochs", type=int, default=45, dest='epochs',
                        help="Specify the maximum number of epochs (default 45)")
    return parser


def train(model, train_loader, criterion, optimizer, losses=None, device=torch.device("cpu")):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_loader):
        images = sample['img']
        labels = sample['expression']
        images.to(device)
        labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if losses is not None:
            running_loss += loss.item()*images.size(0)

    if losses is not None:
        losses.append(running_loss / len(train_loader))


def validate(val_loader, model, criterion, device=torch.device("cpu")):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            images = sample['img']
            labels = sample['expression']
            images.to(device)
            labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total % 1000 == 0:
                print("Accuracy: {} %".format(100*correct/total))
        acc = 100 * correct / total
        print("Accuracy: {} %".format(acc))
        return acc


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_distribution(dataset, title=""):
    hist = [0]*len(Expression)
    for sample in dataset:
        hist[sample['expression']] += 1
    exp = [e.name for e in Expression]
    plt.bar(np.arange(len(exp)), hist)
    plt.title(title)
    plt.xticks(np.arange(len(exp)), exp, rotation=90)
    plt.xlabel("Expression")
    plt.ylabel("Frequency (total count {})".format(len(dataset)))
    plt.show()
