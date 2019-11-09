from torch.utils import data
import data_loader as dl
import neural_nets as nns
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import utils as util


# TODO: Add arg parser to configure what NN is used and assign it a task

net = nns.VggVdFaceFerDag()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             lambda x: x*255,
             transforms.Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1])])

trainset = dl.FER2013Dataset(train=True, tf=transform)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
testset = dl.FER2013Dataset(train=False, tf=transform)
testloader = data.DataLoader(testset, batch_size=4,
                             shuffle=True, num_workers=0)

util.fer2013_test_nn(net, testloader)
