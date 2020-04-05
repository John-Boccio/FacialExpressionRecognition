"""
Author(s):
    John Boccio
Last revision:
    2/25/2020
Description:
    Place for common classes and functions among the FER datasets.

"""
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import torch
import visdom


class FerExpression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6


class FerPlusExpression(Enum):
    NEUTRAL = 0
    HAPPINESS = 1
    SURPRISE = 2
    SADNESS = 3
    ANGER = 4
    DISGUST = 5
    FEAR = 6
    CONTEMPT = 7
    UNKNOWN = 8


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


def show_distribution(dataset, title="", ferplus=False):
    if ferplus:
        hist = [0]*len(FerPlusExpression)
        exp = [e.name for e in FerPlusExpression]
    else:
        hist = [0]*len(FerExpression)
        exp = [e.name for e in FerExpression]

    for sample in dataset:
        hist[sample['expression']] += 1
    plt.bar(np.arange(len(exp)), hist)
    plt.title(title)
    plt.xticks(np.arange(len(exp)), exp, rotation=90)
    plt.xlabel("Expression")
    plt.ylabel("Frequency (total count {})".format(len(dataset)))
    plt.tight_layout()
    plt.savefig(title + ".png")


class VisdomLinePlotter(object):
    """ https://github.com/noagarcia/visdom-tutorial """
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                          name=split_name, update='append')


def graph_losses(train_losses, val_losses, save_path="losses.png"):
    """ https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb """
    plt.plot(range(0, len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(0, len(val_losses)), val_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = val_losses.index(min(val_losses))
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(max(train_losses), max(val_losses)) + 0.25)     # consistent scale
    plt.xlim(0, len(train_losses)+1)                                # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def get_expression(model, img, exp_class=FerExpression):
    model_prediction = model.forward(img.unsqueeze(0))
    _, predicted = torch.max(model_prediction.data, 1)
    expression = exp_class(predicted.item())
    # Return the expression with the greatest probability and the probability distribution
    return expression.name, model_prediction.data
