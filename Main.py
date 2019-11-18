from torch.utils import data
import data_loader as dl
import neural_nets as nns
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import utils as util
import cv2
import os


# TODO: Add arg parser to configure what NN is used and assign it a task

trainset = dl.FER2013Dataset(train=True, tf=None)
image = trainset[0]["img"]
expression = trainset[0]["expression"]
image.show()
print(expression)

device= torch.device('cpu')
if torch.cuda.is_available():
    device= torch.device('cuda')
net = nns.VggVdFaceFerDag()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)


def facechop(image):
    facedata = 'C:\\Users\\Owner\\PycharmProjects\\FacialExpressionRecognition\\image_processing\\haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(facedata)
    ##DIR = 'C:/Users/Owner/PycharmProjects/untitled2/input/'
    pictures = [name for name in os.listdir(image) if os.path.isfile(os.path.join(image, name))]

    ##for pic in pictures:
    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        face_file_name = "C:/Users/Owner/PycharmProjects/FacialExpressionRecognition/image_processing/cropped/" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, sub_face)

    cv2.imshow(image, img)

if __name__ :

    facechop("image")

    while(True):
        key = cv2.waitKey(30)
        if key in [27,
                   ord('Q'), ord('q')]:
            break

img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    face_file_name = "C:/Users/Owner/PycharmProjects/image_processing/normalized/" + ".jpg"
    cv2.imwrite(face_file_name, img_output)
    cv2.imshow('Color input image', image)
    cv2.imshow('Histogram equalized', img_output)

trainset = dl.FER2013Dataset(train=True, tf=img_output)
trainloader = data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=0)
testset = dl.FER2013Dataset(train=False, tf=img_output)
testloader = data.DataLoader(testset, batch_size=4,
                             shuffle=True, num_workers=0)

util.fer2013_test_nn(net, testloader)
