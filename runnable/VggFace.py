"""
Author(s):
    John Boccio
Last revision:
    2/25/2020
Description:

"""
import data_loader as dl
import image_processing
import neural_nets as nns
import torch
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
from utils import FerUtils
from utils import DatasetType


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = nns.VggVdFaceFerDag()
    net.to(device)
    vggface_transform = transforms.Compose(
        [#image_processing.crop_face_transform,
         transforms.Resize(net.meta["imageSize"][0]),
         transforms.ToTensor(),
         lambda x: x*255,
         transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])])

    testset = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=vggface_transform)
    testloader = data.DataLoader(testset, batch_size=4,
                                 shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    FerUtils.validate(testloader, net, criterion, device=device)

