"""
Author(s):
    John Boccio
Last revision:
    10/27/2019
Description:

"""
from torch.utils import data
import data_loader as dl
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


def fer2013_train_nn(model, save_path, epochs=10):
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
