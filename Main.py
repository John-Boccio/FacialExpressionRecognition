from torch.utils import data
from PIL import Image
from utils import DatasetType
import cv2
import data_loader as dl
import image_processing
import neural_nets as nns
import numpy
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import utils


# TODO: Add arg parser to configure what NN is used and assign it a task

net = nns.VggVdFaceFerDag()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
transform = transforms.Compose(
            [image_processing.crop_face_transform,
             transforms.Resize(net.meta["imageSize"][0]),
             transforms.ToTensor(),
             lambda x: x*255,
             transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])])

"""
trainset = dl.FER2013Dataset(set_type=DatasetType.TRAIN, tf=None)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
testset = dl.FER2013Dataset(set_type=DatasetType.TEST, tf=transform)
testloader = data.DataLoader(testset, batch_size=4,
                             shuffle=True, num_workers=0)

utils.fer2013_test_nn(net, testloader)
"""

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = image_processing.crop_faces(pil_frame)

    for f in faces:
        x, y = f["coord"]
        w, h = f["size"]
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        nn_prediction = net.forward(transform(f["img"]).unsqueeze(0))
        _, predicted = torch.max(nn_prediction.data, 1)
        expression = utils.Expression(predicted.item())
        frame = cv2.putText(frame, str(expression), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
