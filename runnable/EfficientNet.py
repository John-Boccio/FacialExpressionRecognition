import data_loader as dl
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils import data
from utils import FerDatasets
from utils import DatasetType


train_transform = transforms.Compose(
    [transforms.Resize(600),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)
trainset = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=train_transform)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=len(FerDatasets.Expression), in_channels=1)

