import data_loader as dl
from efficientnet_pytorch import EfficientNet, utils
import torch
import torch.nn as nn
from utils import FerUtils


class FerEfficientNet(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(FerEfficientNet, self).__init__()
        self.conv_net = EfficientNet.from_pretrained('efficientnet-b4',
                                                     num_classes=len(FerUtils.Expression),
                                                     in_channels=3)
        self.conv_net.to(device)
        # We will keep the conv layers and create new fully connected layers for transfer learning on FER
        for param in self.conv_net.parameters():
            param.requires_grad = False

        # Conv layers output [1, 2560, 18, 18] for efficientnet-b4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout2d(p=0.9)
        self.fc1 = nn.Linear(in_features=2560, out_features=1000, bias=True)
        self.fc1_swish = utils.MemoryEfficientSwish()
        self.dropout2 = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(in_features=1000, out_features=len(FerUtils.Expression), bias=True)
        self.fc2_swish = utils.MemoryEfficientSwish()

    def forward(self, inputs):
        # Conv layers --> outputs [1, 2560, 18, 18]
        x = self.conv_net.extract_features(inputs)
        x = self.avg_pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 2560)
        x = self.fc1(x)
        x = self.fc1_swish(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc2_swish(x)
        return x
