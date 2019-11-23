from torch.utils import data
from utils import DatasetType
import data_loader as dl
import image_processing
import neural_nets as nns
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import utils


# TODO: Add arg parser to configure what NN is used and assign it a task

net = nns.VggVdFaceFerDag()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
transform = transforms.Compose(
            [lambda x: image_processing.crop_faces(x, one_face=True),
             transforms.Resize(net.meta["imageSize"][0]),
             transforms.ToTensor(),
             lambda x: x*255,
             transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])])

trainset = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=None)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
testset = dl.FER2013Dataset(set_type=DatasetType.TEST, tf=transform)
testloader = data.DataLoader(testset, batch_size=4,
                             shuffle=True, num_workers=0)

utils.fer2013_test_nn(net, testloader)
