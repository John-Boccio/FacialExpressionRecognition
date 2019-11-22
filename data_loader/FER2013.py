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
from PIL import Image
from utils import DatasetType
import ConfigParser as Cp
import csv
import numpy as np
import os
import pickle
import torch
import warnings


class FER2013Dataset(Dataset):
    __train = None
    __test = None
    __validation = None

    def __init__(self, set_type=DatasetType.TRAIN, tf=None):
        self.transform = tf
        self.set_type = set_type

        # Check if the dataset has already been initialized
        if FER2013Dataset.__train is not None \
                and FER2013Dataset.__test is not None \
                and FER2013Dataset.__validation is not None:
            return

        # If dataset is not initialized, check if we have it pickled
        if os.path.exists("./metadata/fer2013/fer2013.pickle"):
            fer2013 = pickle.load(open("./metadata/fer2013/fer2013.pickle", "rb"))
            FER2013Dataset.__train = fer2013["train"]
            FER2013Dataset.__test = fer2013["test"]
            FER2013Dataset.__validation = fer2013["validation"]
            return

        # Initialize it the hard way
        FER2013Dataset.__train = []
        FER2013Dataset.__test = []
        FER2013Dataset.__validation = []

        fer2013_config = Cp.ConfigParser.get_config()["data_loader"]["FER2013"]
        # Read CSV containing images, labels, and set type
        with open(fer2013_config["csv_path"]) as csv_file:
            # Remove first row since it just says the ordering of data
            csv_file.readline()

            csv_reader = csv.reader(csv_file, delimiter=',')
            for entry in csv_reader:
                emotion = int(entry[0])
                pixels = entry[1]

                # Convert string of pixel values to image
                pixels = [int(pixel) for pixel in pixels.split(' ')]
                pixels = np.asarray(pixels).reshape((48, 48))
                pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)

                # Create data point
                data_point = {
                    "img": np.asarray(pixels).astype('uint8'),
                    # The emotion labels follow the Expression enum exactly
                    "expression": emotion
                }
                if entry[2] == "Training":
                    FER2013Dataset.__train.append(data_point)
                elif entry[2] == "PublicTest":
                    FER2013Dataset.__test.append(data_point)
                elif entry[2] == "PrivateTest":
                    FER2013Dataset.__test.append(data_point)
                else:
                    warnings.warn("Unknown dataset type: " + entry[2])

        dump = {"train":        FER2013Dataset.__train,
                "test":         FER2013Dataset.__test,
                "validation":   FER2013Dataset.__validation}
        pickle.dump(dump, open("./metadata/fer2013/fer2013.pickle", "wb"))

    def __len__(self):
        if self.set_type == DatasetType.TRAIN:
            return len(FER2013Dataset.__train)
        elif self.set_type == DatasetType.TEST:
            return len(FER2013Dataset.__test)
        elif self.set_type == DatasetType.VALIDATION:
            return len(FER2013Dataset.__validation)
        return -1

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Deep copies so the user can't mess with the dataset
        if self.set_type == DatasetType.TRAIN:
            sample = FER2013Dataset.__train[item].copy()
        elif self.set_type == DatasetType.TEST:
            sample = FER2013Dataset.__test[item].copy()
        elif self.set_type == DatasetType.VALIDATION:
            sample = FER2013Dataset.__validation[item].copy()
        else:
            return None

        sample["img"] = Image.fromarray(sample["img"])
        if self.transform:
            sample["img"] = self.transform(sample["img"])
        return sample
