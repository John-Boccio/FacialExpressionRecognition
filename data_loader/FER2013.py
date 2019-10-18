"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:
    API for using the "FER2013" dataset. To use this, you must have the paths defined in the config.json. This API
    splits the data into a training set (data labeled "Training") and a test set (data labeled "PublicTest" or
    "PrivateTest").
    label：(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
"""
from torch.utils.data import Dataset
from skimage import io, transform
import ConfigParser as Cp
import csv
import numpy as np
import os
import pickle
import torch


class FER2013Dataset(Dataset):
    __train = None
    __test = None

    def __init__(self, train=True, tf=None):
        self.transform = tf
        self.train = train

        # Check if the dataset has already been initialized
        if FER2013Dataset.__train is not None and FER2013Dataset.__test is not None:
            return

        # If dataset is not initialized, check if we have it pickled
        should_pickle = Cp.ConfigParser.get_config()["data_loader"]["pickle"]
        if should_pickle and os.path.exists("./metadata/fer2013/train.pickle") and \
                os.path.exists("./metadata/fer2013/test.pickle"):
            FER2013Dataset.__train = pickle.load(open("./metadata/fer2013/train.pickle", "rb"))
            FER2013Dataset.__test = pickle.load(open("./metadata/fer2013/test.pickle", "rb"))
            return

        # Initialize it the hard way
        FER2013Dataset.__train = []
        FER2013Dataset.__test = []

        fer2013_config = Cp.ConfigParser.get_config()["data_loader"]["FER2013"]
        # Read CSV containing images, labels, and set type
        with open(fer2013_config["csv_path"]) as csv_file:
            # Remove first row since it just says the ordering of data
            csv_file.readline()

            csv_reader = csv.reader(csv_file, delimiter=',')
            for entry in csv_reader:
                training_set = True if entry[2] == "Training" else False
                if training_set != train:
                    continue

                emotion = entry[0]
                pixels = entry[1]

                # Convert string of pixel values to image
                pixels = [int(pixel) for pixel in pixels.split(' ')]
                pixels = np.asarray(pixels).reshape((48, 48))

                # Create data point
                data_point = {
                    "img":      pixels,
                    # The emotion labels follow the Expression.py enum exactly
                    "expression":  emotion
                }
                if entry[2] == "Training":
                    FER2013Dataset.__train.append(data_point)
                else:
                    FER2013Dataset.__test.append(data_point)

        if should_pickle:
            pickle.dump(FER2013Dataset.__train, open("./metadata/fer2013/train.pickle", "wb"))
            pickle.dump(FER2013Dataset.__test, open("./metadata/fer2013/test.pickle", "wb"))

    def __len__(self):
        if self.train:
            return len(FER2013Dataset.__train)
        else:
            return len(FER2013Dataset.__test)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.train:
            sample = FER2013Dataset.__train[item]
        else:
            sample = FER2013Dataset.__test[item]

        if self.transform:
            sample = self.transform(sample["img"])

        return sample
