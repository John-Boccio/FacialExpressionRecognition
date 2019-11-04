"""
Author(s):
    John Boccio
Last revision:
    10/27/2019
Description:

"""
from torch.utils import data
import FacialExpressionRecognition.data_loader as dl
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


def fer2013_train_nn(model, save_path, epochs=2):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in enumerate(model.trainloader):
            inputs = sample['img']
            labels = sample['expression']
            model.optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()
            if i % 3000 == 2999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 3000))
                running_loss = 0.0

    torch.save(model.state_dict(), save_path)


def fer2013_test_nn(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in model.testloader:
            inputs = sample['img']
            labels = sample['expression']
            nn_prediction = model.forward(inputs)
            _, predicted = torch.max(nn_prediction.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy: {} %".format(100*correct/total))


class Fer2013V1(nn.Module):
    def __init__(self, criterion=None, optimizer=None, tf=None):
        super(Fer2013V1, self).__init__()

        self.trainset = dl.FER2013Dataset(train=True, tf=tf)
        self.trainloader = data.DataLoader(self.trainset, batch_size=4,
                                           shuffle=True, num_workers=0)
        self.testset = dl.FER2013Dataset(train=False, tf=tf)
        self.testloader = data.DataLoader(self.testset, batch_size=4,
                                          shuffle=True, num_workers=0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)   # 1x48x48 -> 32x46x46
        self.pool1 = nn.MaxPool2d(2, 2)         # 32x46x46 -> 32x23x23
        self.conv2 = nn.Conv2d(32, 85, 2)       # 32x23x23 -> 85x22x22
        self.pool2 = nn.MaxPool2d(2, 2)         # 85x22x22 -> 85x11x11
        self.conv3 = nn.Conv2d(85, 128, 3)       # 85x11x11 -> 128x9x9
        self.conv4 = nn.Conv2d(128, 128, 3)       # 128x9x9 -> 128x7x7
        self.conv5 = nn.Conv2d(128, 85, 3)       # 128x7x7 -> 85x5x5
        self.fc1 = nn.Linear(85*5*5, 1024)      # 2125 -> 1024
        self.fc2 = nn.Linear(1024, 512)         # 1024 -> 512
        self.fc3 = nn.Linear(512, 7)            # 512 -> 7

        if os.path.exists("metadata/neural_nets/Fer2013V1.torch"):
            self.load_state_dict(torch.load("metadata/neural_nets/fer2013V1.torch"))

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        else:
            self.optimizer = optimizer

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.pool2(x)
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = x.view(-1, 85*5*5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=1)
        return x
