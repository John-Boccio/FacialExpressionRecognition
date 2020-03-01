"""
Author(s):
    John Boccio
Last revision:
    2/25/2020
Description:
    Place for common classes and functions among the FER datasets.

    A large portion of this has been taken from PyTorch's ImageNet example and changed for FER
    (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
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
