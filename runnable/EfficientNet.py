import data_loader as dl
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_nets import FerEfficientNet
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils import data
from utils import FerDatasets
from utils import DatasetType


def train(model, train_loader, criterion, optimizer, device="cpu"):
    for i, sample in enumerate(train_loader):
        batch = sample['img']
        labels = sample['expression']
        batch.to(device)
        labels.to(device)

        output = model(batch)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_transform = transforms.Compose(
    [transforms.Resize(600),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)
trainset = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=train_transform)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FerEfficientNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=.01)
model.to(device)
criterion.to(device)

