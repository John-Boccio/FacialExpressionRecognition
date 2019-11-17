"""
Author(s):
    John Boccio
Last revision:
    10/27/2019
Description:
    Place for common classes and functions among the FER datasets.
"""
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class Expression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6


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
