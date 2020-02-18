"""
Author(s):
    John Boccio
Last revision:
    10/27/2019
Description:

"""
import data_loader as dl
import image_processing
import neural_nets as nns
import torch
from torchvision import transforms
from torch.utils import data
from utils import FerDatasets
from utils import DatasetType


def vggface_accuracy_benchmark(model, loader, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in loader:
            sample.to(device)
            inputs = sample['img']
            labels = sample['expression']
            nn_prediction = model.forward(inputs)
            _, predicted = torch.max(nn_prediction.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total % 1000 == 0:
                print("Accuracy: {} %".format(100*correct/total))
    print("Accuracy: {} %".format(100*correct/total))


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

vggface_accuracy_benchmark(net, testloader)

