import argparse
from datetime import datetime
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
from utils import FerUtils
from utils import DatasetType


def main():
    args = FerUtils.get_fer_parser().parse_args()
    print("Setting up efficient net...")
    print(" - Fetching FER dataset")
    train_transform = transforms.Compose(
        [transforms.Resize(600),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    train_set = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=train_transform)
    val_set = dl.FER2013Dataset(set_type=DatasetType.VALIDATION, tf=train_transform)
    train_loader = data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" - Initializing on device: ", device)

    print(" - Initializing FER EfficientNet")
    model = FerEfficientNet(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model.to(device)
    criterion.to(device)
    print()

    if args.eval:
        print("Performing evaluation of FER EfficientNet using FER2013 validation set...")
        accuracy = FerUtils.validate(val_loader, model, criterion, device=device)
        print("Accuracy: {} %", accuracy)
        return

    start_time = datetime.now()
    losses = []
    print("Beginning training, current time: ", start_time)
    for epoch in range(args.epochs):
        FerUtils.adjust_learning_rate(optimizer, epoch, args.lr)
        FerUtils.train(model, train_loader, criterion, optimizer, losses=losses, device=device)
        accuracy = FerUtils.validate(val_loader, model, criterion)


if __name__ == "__main__":
    main()
