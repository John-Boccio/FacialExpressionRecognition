"""
Author(s):
    John Boccio
Last revision:
    11/8/2019
Description:
    Version 1 of a custom CNN used for FER2013
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as f


class Fer2013V1(nn.Module):
    def __init__(self):
        super(Fer2013V1, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)        # 48x48 -> 46x46
        self.conv2 = nn.Conv2d(32, 32, 3)       # 46x46 -> 44x44
        self.pool1 = nn.MaxPool2d(2, 2)         # 44x44 -> 22x22
        self.conv3 = nn.Conv2d(32, 64, 3)       # 22x22 -> 20x20
        self.conv4 = nn.Conv2d(64, 64, 3)       # 20x20 -> 18x18
        self.conv5 = nn.Conv2d(64, 96, 3)       # 18x18 -> 16x16
        self.pool2 = nn.MaxPool2d(2, 2)         # 16x16 -> 8x8
        self.fc1 = nn.Linear(96*8*8, 1024)      # 6144 -> 1024
        self.fc2 = nn.Linear(1024, 512)         # 1024 -> 512
        self.fc3 = nn.Linear(512, 7)            # 512 -> 7

        if os.path.exists("metadata/neural_nets/Fer2013V1.torch"):
            self.load_state_dict(torch.load("metadata/neural_nets/fer2013V1.torch"))

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool1(x)
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = self.pool2(x)
        x = x.view(-1, 96*8*8)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=1)
        return x
